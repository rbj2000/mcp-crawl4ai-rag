"""Ollama AI provider implementation"""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientSession

from ..base import (
    EmbeddingProvider, 
    LLMProvider, 
    HybridAIProvider,
    RerankingProvider,
    EmbeddingResult, 
    LLMResult, 
    HealthStatus,
    RerankingResult,
    AIProvider
)

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider implementation"""
    
    # Common Ollama embedding models with their approximate dimensions
    MODEL_CONFIGS = {
        "mxbai-embed-large": {"dimensions": 1024, "size": "669M", "family": "bert"},
        "nomic-embed-text": {"dimensions": 768, "size": "274M", "family": "bert"},
        "all-minilm": {"dimensions": 384, "size": "46M", "family": "bert"},
        "snowflake-arctic-embed": {"dimensions": 1024, "size": "669M", "family": "bert"},
        "bge-large": {"dimensions": 1024, "size": "1.34GB", "family": "bert"},
        "bge-base": {"dimensions": 768, "size": "438M", "family": "bert"},
        "gte-large": {"dimensions": 1024, "size": "670M", "family": "bert"},
        "gte-base": {"dimensions": 768, "size": "219M", "family": "bert"},
        "e5-large": {"dimensions": 1024, "size": "1.34GB", "family": "bert"},
        "e5-base": {"dimensions": 768, "size": "438M", "family": "bert"},
        "multilingual-e5-large": {"dimensions": 1024, "size": "2.24GB", "family": "bert"},
    }
    
    # Ollama reranking models (these are separate from embedding models)
    RERANKING_MODEL_CONFIGS = {
        "bge-reranker-base": {"size": "278M", "description": "BGE reranking model - base version"},
        "bge-reranker-large": {"size": "560M", "description": "BGE reranking model - large version"},
        "ms-marco-MiniLM-L-6-v2": {"size": "22.7M", "description": "MS MARCO MiniLM reranker"},
        "bge-reranker-v2-m3": {"size": "568M", "description": "BGE reranker v2 multilingual"},
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama embedding provider
        
        Args:
            config: Configuration dictionary with Ollama settings
        """
        super().__init__(config)
        self.session: Optional[ClientSession] = None
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("embedding_model", self.default_model)
        
        # Connection settings
        self.timeout = config.get("timeout", 120.0)  # Ollama can be slow for large models
        self.connect_timeout = config.get("connect_timeout", 10.0)
        
        # Rate limiting and retry settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 2.0)
        self.max_retry_delay = config.get("max_retry_delay", 60.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)
        
        # Batch processing settings (Ollama typically handles one at a time efficiently)
        self._batch_size = config.get("batch_size", 10)
        
        # Model management
        self.auto_pull_models = config.get("auto_pull_models", True)
        self.pull_timeout = config.get("pull_timeout", 600.0)  # 10 minutes for model downloads
        
        # Performance monitoring
        self._request_count = 0
        self._total_texts_processed = 0
        self._error_count = 0
        self._model_pull_count = 0
        
        # Cache for model dimensions (auto-detected)
        self._dimension_cache: Dict[str, int] = {}
        
    def _validate_config(self) -> None:
        """Validate Ollama-specific configuration"""
        base_url = self.config.get("base_url", "http://localhost:11434")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url '{base_url}'. Must start with http:// or https://")
        
        # Validate timeout values
        timeout = self.config.get("timeout", 120.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number")
        
        # Warn about model if not in known configs
        model = self.config.get("embedding_model")
        if model and model not in self.MODEL_CONFIGS:
            logger.warning(
                f"Model '{model}' not in known configurations. "
                f"Supported models: {list(self.MODEL_CONFIGS.keys())}"
            )
    
    async def initialize(self) -> None:
        """Initialize the Ollama HTTP client"""
        try:
            # Create HTTP session with timeouts
            timeout = ClientTimeout(
                total=self.timeout,
                connect=self.connect_timeout
            )
            
            self.session = ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
            
            # Test connection and ensure model is available
            await self._ensure_model_available(self.model)
            
            # Perform health check
            health = await self.health_check()
            if not health.is_healthy:
                raise ConnectionError(f"Ollama service health check failed: {health.error_message}")
            
            logger.info(f"Ollama embedding provider initialized successfully with model '{self.model}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding provider: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise ConnectionError(f"Failed to initialize Ollama client: {e}")
    
    async def close(self) -> None:
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Ollama embedding provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Check if Ollama service is healthy and model is available"""
        if not self.session:
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                response_time_ms=0.0,
                error_message="Session not initialized"
            )
        
        start_time = time.time()
        try:
            # Check if Ollama service is running
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    response_time_ms = (time.time() - start_time) * 1000
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Ollama service returned status {response.status}"
                    )
                
                models_data = await response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                # Check if our model is available
                model_available = any(
                    self.model in model_name or model_name.startswith(self.model + ":")
                    for model_name in available_models
                )
                
                if not model_available:
                    response_time_ms = (time.time() - start_time) * 1000
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Model '{self.model}' not available. Available models: {available_models}"
                    )
            
            # Test with a simple embedding request
            test_payload = {
                "model": self.model,
                "prompt": "test"
            }
            
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=test_payload
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "embedding" in data and data["embedding"]:
                        return HealthStatus(
                            is_healthy=True,
                            provider=AIProvider.OLLAMA.value,
                            model=self.model,
                            response_time_ms=response_time_ms,
                            metadata={
                                "embedding_dimensions": len(data["embedding"]),
                                "available_models": available_models
                            }
                        )
                    else:
                        return HealthStatus(
                            is_healthy=False,
                            provider=AIProvider.OLLAMA.value,
                            model=self.model,
                            response_time_ms=response_time_ms,
                            error_message="Empty embedding response"
                        )
                else:
                    error_text = await response.text()
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Embedding test failed: {response.status} - {error_text}"
                    )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """Create an embedding for a single text"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        use_model = model or self.model
        
        # Check text length
        if len(text) > self.max_text_length:
            logger.warning(f"Text length {len(text)} exceeds maximum {self.max_text_length}, truncating")
            text = text[:self.max_text_length]
        
        try:
            result = await self.create_embeddings_batch([text], model=use_model)
            return result[0]
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise
    
    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[EmbeddingResult]:
        """Create embeddings for multiple texts"""
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and log warnings
        valid_texts = []
        empty_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.warning(f"Empty text at index {i}, will return zero embedding")
                empty_indices.append(i)
                valid_texts.append("")  # Placeholder
            else:
                # Truncate if too long
                if len(text) > self.max_text_length:
                    logger.warning(f"Text at index {i} length {len(text)} exceeds maximum, truncating")
                    text = text[:self.max_text_length]
                valid_texts.append(text)
        
        use_model = model or self.model
        use_batch_size = min(batch_size or self._batch_size, self.max_batch_size)
        
        # Ensure model is available
        await self._ensure_model_available(use_model)
        
        # Process texts (Ollama typically processes one at a time efficiently)
        all_results = []
        
        for i in range(0, len(valid_texts), use_batch_size):
            batch_end = min(i + use_batch_size, len(valid_texts))
            batch_texts = valid_texts[i:batch_end]
            
            try:
                # Process each text in the batch
                batch_results = []
                for text in batch_texts:
                    if text:  # Non-empty text
                        result = await self._create_single_embedding_with_retry(text, use_model)
                        batch_results.append(result)
                    else:  # Empty text placeholder
                        dimensions = await self._get_model_dimensions(use_model)
                        batch_results.append(EmbeddingResult(
                            embedding=[0.0] * dimensions,
                            dimensions=dimensions,
                            model=use_model,
                            input_tokens=0
                        ))
                
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//use_batch_size + 1}: {e}")
                
                # Create zero embeddings as fallback
                dimensions = await self._get_model_dimensions(use_model)
                for _ in batch_texts:
                    all_results.append(EmbeddingResult(
                        embedding=[0.0] * dimensions,
                        dimensions=dimensions,
                        model=use_model,
                        input_tokens=0
                    ))
        
        return all_results
    
    async def _create_single_embedding_with_retry(self, text: str, model: str) -> EmbeddingResult:
        """Create a single embedding with retry logic"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                payload = {
                    "model": model,
                    "prompt": text
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "embedding" not in data or not data["embedding"]:
                            raise ValueError("Empty embedding response from Ollama")
                        
                        embedding = data["embedding"]
                        dimensions = len(embedding)
                        
                        # Cache dimensions for this model
                        self._dimension_cache[model] = dimensions
                        
                        # Update metrics
                        self._request_count += 1
                        self._total_texts_processed += 1
                        
                        result = EmbeddingResult(
                            embedding=embedding,
                            dimensions=dimensions,
                            model=model,
                            input_tokens=len(text.split())  # Rough token estimate
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        logger.debug(
                            f"Created embedding for {len(text)} chars in {processing_time:.1f}ms "
                            f"(attempt {attempt + 1})"
                        )
                        
                        return result
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                self._error_count += 1
                
                if attempt < self.max_retries:
                    if self._is_retryable_error(e):
                        logger.warning(
                            f"Embedding request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {retry_delay:.1f}s"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(
                            retry_delay * self.retry_exponential_base,
                            self.max_retry_delay
                        )
                        continue
                    else:
                        logger.error(f"Non-retryable error in embedding request: {e}")
                        raise
                else:
                    logger.error(f"Embedding request failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    async def _ensure_model_available(self, model: str) -> None:
        """Ensure the specified model is available, pull if necessary"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        try:
            # Check if model is already available
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    
                    # Check if model is available (exact match or with tag)
                    if any(model in m or m.startswith(model + ":") for m in available_models):
                        return
                    
                    logger.info(f"Model '{model}' not found locally. Available models: {available_models}")
                    
                    if self.auto_pull_models:
                        await self._pull_model(model)
                    else:
                        raise ValueError(
                            f"Model '{model}' not available and auto_pull_models is disabled. "
                            f"Available models: {available_models}"
                        )
                else:
                    raise Exception(f"Failed to check available models: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            raise
    
    async def _pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        logger.info(f"Pulling model '{model}' from Ollama registry...")
        start_time = time.time()
        
        try:
            payload = {"name": model}
            
            # Use a longer timeout for model pulling
            pull_timeout = ClientTimeout(total=self.pull_timeout)
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=pull_timeout
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                progress_data = json.loads(line.decode().strip())
                                if "status" in progress_data:
                                    status = progress_data["status"]
                                    if "completed" in progress_data and "total" in progress_data:
                                        completed = progress_data["completed"]
                                        total = progress_data["total"]
                                        percent = (completed / total) * 100 if total > 0 else 0
                                        logger.info(f"Pulling {model}: {status} ({percent:.1f}%)")
                                    else:
                                        logger.info(f"Pulling {model}: {status}")
                            except json.JSONDecodeError:
                                continue
                    
                    pull_time = time.time() - start_time
                    self._model_pull_count += 1
                    logger.info(f"Successfully pulled model '{model}' in {pull_time:.1f}s")
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to pull model: HTTP {response.status} - {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Model pull timed out after {self.pull_timeout}s")
        except Exception as e:
            logger.error(f"Failed to pull model '{model}': {e}")
            raise
    
    async def _get_model_dimensions(self, model: str) -> int:
        """Get or detect the embedding dimensions for a model"""
        # Check cache first
        if model in self._dimension_cache:
            return self._dimension_cache[model]
        
        # Check known configurations
        if model in self.MODEL_CONFIGS:
            dimensions = self.MODEL_CONFIGS[model]["dimensions"]
            self._dimension_cache[model] = dimensions
            return dimensions
        
        # Auto-detect by creating a test embedding
        try:
            logger.info(f"Auto-detecting dimensions for model '{model}'")
            test_result = await self._create_single_embedding_with_retry("test", model)
            return test_result.dimensions
        except Exception as e:
            logger.warning(f"Failed to auto-detect dimensions for '{model}': {e}")
            # Return a reasonable default
            return 768
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()
        
        # Connection errors
        if any(term in error_str for term in ["connection", "timeout", "network"]):
            return True
        
        # HTTP server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True
        
        # Ollama specific errors
        if "model not found" in error_str:
            return False  # Don't retry if model doesn't exist
        
        # Rate limiting (unlikely with local Ollama but possible)
        if "rate limit" in error_str or "429" in error_str:
            return True
        
        # aiohttp specific errors
        if isinstance(error, (ClientError, asyncio.TimeoutError)):
            return True
        
        return False
    
    @property
    def default_model(self) -> str:
        """Default embedding model"""
        return "mxbai-embed-large"
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported embedding models"""
        return list(self.MODEL_CONFIGS.keys())
    
    def get_model_dimensions(self, model: str) -> int:
        """Get embedding dimensions for a model"""
        if model in self._dimension_cache:
            return self._dimension_cache[model]
        
        if model in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model]["dimensions"]
        
        # Default for unknown models
        logger.warning(f"Unknown model '{model}', using default dimensions")
        return 768
    
    @property
    def max_batch_size(self) -> int:
        """Maximum batch size for Ollama embeddings"""
        return self._batch_size
    
    @property
    def max_text_length(self) -> int:
        """Maximum text length for Ollama embeddings"""
        # Ollama models generally handle longer texts well
        return 32768
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics"""
        return {
            "request_count": self._request_count,
            "total_texts_processed": self._total_texts_processed,
            "error_count": self._error_count,
            "model_pull_count": self._model_pull_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "model": self.model,
            "dimension_cache": dict(self._dimension_cache)
        }


class OllamaLLMProvider(LLMProvider):
    """Ollama LLM provider implementation"""
    
    # Common Ollama LLM models with their configurations
    MODEL_CONFIGS = {
        # Llama models
        "llama2": {"family": "llama", "size": "3.8GB", "context_length": 4096},
        "llama2:13b": {"family": "llama", "size": "7.3GB", "context_length": 4096},
        "llama2:70b": {"family": "llama", "size": "39GB", "context_length": 4096},
        "llama3": {"family": "llama", "size": "4.7GB", "context_length": 8192},
        "llama3:8b": {"family": "llama", "size": "4.7GB", "context_length": 8192},
        "llama3:70b": {"family": "llama", "size": "40GB", "context_length": 8192},
        "llama3.1": {"family": "llama", "size": "4.7GB", "context_length": 131072},
        "llama3.1:8b": {"family": "llama", "size": "4.7GB", "context_length": 131072},
        "llama3.1:70b": {"family": "llama", "size": "40GB", "context_length": 131072},
        "llama3.1:405b": {"family": "llama", "size": "229GB", "context_length": 131072},
        
        # Code-specialized models
        "codellama": {"family": "llama", "size": "3.8GB", "context_length": 16384},
        "codellama:13b": {"family": "llama", "size": "7.3GB", "context_length": 16384},
        "codellama:34b": {"family": "llama", "size": "19GB", "context_length": 16384},
        "codeqwen": {"family": "qwen", "size": "4.2GB", "context_length": 65536},
        
        # Mistral models
        "mistral": {"family": "mistral", "size": "4.1GB", "context_length": 32768},
        "mistral:7b": {"family": "mistral", "size": "4.1GB", "context_length": 32768},
        "mixtral": {"family": "mistral", "size": "26GB", "context_length": 32768},
        "mixtral:8x7b": {"family": "mistral", "size": "26GB", "context_length": 32768},
        
        # Gemma models
        "gemma": {"family": "gemma", "size": "5.0GB", "context_length": 8192},
        "gemma:2b": {"family": "gemma", "size": "1.4GB", "context_length": 8192},
        "gemma:7b": {"family": "gemma", "size": "5.0GB", "context_length": 8192},
        "gemma2": {"family": "gemma", "size": "5.4GB", "context_length": 8192},
        "gemma2:9b": {"family": "gemma", "size": "5.4GB", "context_length": 8192},
        "gemma2:27b": {"family": "gemma", "size": "16GB", "context_length": 8192},
        
        # Qwen models
        "qwen": {"family": "qwen", "size": "4.2GB", "context_length": 32768},
        "qwen:7b": {"family": "qwen", "size": "4.2GB", "context_length": 32768},
        "qwen:14b": {"family": "qwen", "size": "8.2GB", "context_length": 32768},
        "qwen2": {"family": "qwen", "size": "4.4GB", "context_length": 131072},
        "qwen2:7b": {"family": "qwen", "size": "4.4GB", "context_length": 131072},
        
        # Phi models
        "phi3": {"family": "phi", "size": "2.3GB", "context_length": 128000},
        "phi3:mini": {"family": "phi", "size": "2.3GB", "context_length": 128000},
        "phi3:medium": {"family": "phi", "size": "7.9GB", "context_length": 128000},
        
        # Other models
        "vicuna": {"family": "llama", "size": "3.8GB", "context_length": 2048},
        "orca-mini": {"family": "llama", "size": "1.9GB", "context_length": 2048},
        "neural-chat": {"family": "mistral", "size": "4.1GB", "context_length": 8192}
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama LLM provider"""
        super().__init__(config)
        self.session: Optional[ClientSession] = None
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("llm_model", self.default_model)
        
        # Connection settings
        self.timeout = config.get("timeout", 300.0)  # 5 minutes for LLM generation
        self.connect_timeout = config.get("connect_timeout", 10.0)
        
        # Generation settings
        self._temperature = config.get("temperature", self.default_temperature)
        self._max_tokens = config.get("max_tokens", self.default_max_tokens)
        
        # Rate limiting and retry settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 2.0)
        self.max_retry_delay = config.get("max_retry_delay", 60.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)
        
        # Model management
        self.auto_pull_models = config.get("auto_pull_models", True)
        self.pull_timeout = config.get("pull_timeout", 600.0)
        
        # Performance monitoring
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._error_count = 0
        self._model_pull_count = 0
    
    def _validate_config(self) -> None:
        """Validate Ollama-specific configuration"""
        base_url = self.config.get("base_url", "http://localhost:11434")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid base_url '{base_url}'. Must start with http:// or https://")
        
        # Validate timeout values
        timeout = self.config.get("timeout", 300.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number")
        
        # Warn about model if not in known configs
        model = self.config.get("llm_model")
        if model and model not in self.MODEL_CONFIGS:
            logger.warning(
                f"Model '{model}' not in known configurations. "
                f"Supported models: {list(self.MODEL_CONFIGS.keys())}"
            )
    
    async def initialize(self) -> None:
        """Initialize the Ollama HTTP client"""
        try:
            # Create HTTP session with timeouts
            timeout = ClientTimeout(
                total=self.timeout,
                connect=self.connect_timeout
            )
            
            self.session = ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
            )
            
            # Test connection and ensure model is available
            await self._ensure_model_available(self.model)
            
            # Perform health check
            health = await self.health_check()
            if not health.is_healthy:
                raise ConnectionError(f"Ollama service health check failed: {health.error_message}")
            
            logger.info(f"Ollama LLM provider initialized successfully with model '{self.model}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM provider: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise ConnectionError(f"Failed to initialize Ollama client: {e}")
    
    async def close(self) -> None:
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Ollama LLM provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Check if Ollama service is healthy and model is available"""
        if not self.session:
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                response_time_ms=0.0,
                error_message="Session not initialized"
            )
        
        start_time = time.time()
        try:
            # Check if Ollama service is running
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    response_time_ms = (time.time() - start_time) * 1000
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Ollama service returned status {response.status}"
                    )
                
                models_data = await response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]
                
                # Check if our model is available
                model_available = any(
                    self.model in model_name or model_name.startswith(self.model + ":")
                    for model_name in available_models
                )
                
                if not model_available:
                    response_time_ms = (time.time() - start_time) * 1000
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Model '{self.model}' not available. Available models: {available_models}"
                    )
            
            # Test with a simple generation request
            test_payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
                "options": {"num_predict": 5}  # Limit tokens for health check
            }
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=test_payload
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    if "message" in data and "content" in data["message"]:
                        return HealthStatus(
                            is_healthy=True,
                            provider=AIProvider.OLLAMA.value,
                            model=self.model,
                            response_time_ms=response_time_ms,
                            metadata={
                                "available_models": available_models,
                                "response_length": len(data["message"]["content"])
                            }
                        )
                    else:
                        return HealthStatus(
                            is_healthy=False,
                            provider=AIProvider.OLLAMA.value,
                            model=self.model,
                            response_time_ms=response_time_ms,
                            error_message="Empty generation response"
                        )
                else:
                    error_text = await response.text()
                    return HealthStatus(
                        is_healthy=False,
                        provider=AIProvider.OLLAMA.value,
                        model=self.model,
                        response_time_ms=response_time_ms,
                        error_message=f"Generation test failed: {response.status} - {error_text}"
                    )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OLLAMA.value,
                model=self.model,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        """Generate text using Ollama chat completion"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        # Validate messages
        self.validate_messages(messages)
        
        use_model = model or self.model
        use_temperature = temperature if temperature is not None else self._temperature
        use_max_tokens = max_tokens or self._max_tokens
        
        # Ensure model is available
        await self._ensure_model_available(use_model)
        
        # Prepare request parameters
        options = {
            "temperature": use_temperature,
            "num_predict": use_max_tokens,
            **kwargs
        }
        
        request_params = {
            "model": use_model,
            "messages": messages,
            "stream": False,
            "options": options
        }
        
        try:
            result = await self._generate_with_retry(request_params)
            return result
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[LLMResult, None]:
        """Generate text with streaming response"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        # Validate messages
        self.validate_messages(messages)
        
        use_model = model or self.model
        use_temperature = temperature if temperature is not None else self._temperature
        use_max_tokens = max_tokens or self._max_tokens
        
        # Ensure model is available
        await self._ensure_model_available(use_model)
        
        # Prepare request parameters
        options = {
            "temperature": use_temperature,
            "num_predict": use_max_tokens,
            **kwargs
        }
        
        request_params = {
            "model": use_model,
            "messages": messages,
            "stream": True,
            "options": options
        }
        
        try:
            async for chunk in self._generate_stream_with_retry(request_params):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to generate streaming text: {e}")
            raise
    
    async def _generate_with_retry(self, request_params: Dict[str, Any]) -> LLMResult:
        """Generate text with retry logic"""
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                async with self.session.post(
                    f"{self.base_url}/api/chat",
                    json=request_params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "message" not in data or "content" not in data["message"]:
                            raise ValueError("Invalid response format from Ollama")
                        
                        content = data["message"]["content"]
                        
                        # Update metrics
                        self._request_count += 1
                        
                        # Estimate tokens (rough approximation)
                        input_tokens = sum(len(msg["content"].split()) for msg in request_params["messages"])
                        output_tokens = len(content.split())
                        
                        self._total_input_tokens += input_tokens
                        self._total_output_tokens += output_tokens
                        
                        result = LLMResult(
                            content=content,
                            model=request_params["model"],
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            finish_reason=data.get("done_reason", "stop"),
                            metadata={
                                "response_time_ms": (time.time() - start_time) * 1000,
                                "eval_count": data.get("eval_count"),
                                "eval_duration": data.get("eval_duration"),
                                "load_duration": data.get("load_duration"),
                                "prompt_eval_count": data.get("prompt_eval_count"),
                                "prompt_eval_duration": data.get("prompt_eval_duration"),
                                "total_duration": data.get("total_duration")
                            }
                        )
                        
                        logger.debug(
                            f"Generated {len(content)} characters in "
                            f"{result.metadata['response_time_ms']:.1f}ms (attempt {attempt + 1})"
                        )
                        
                        return result
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                        
            except Exception as e:
                self._error_count += 1
                
                if attempt < self.max_retries:
                    if self._is_retryable_error(e):
                        logger.warning(
                            f"LLM request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {retry_delay:.1f}s"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(
                            retry_delay * self.retry_exponential_base,
                            self.max_retry_delay
                        )
                        continue
                    else:
                        logger.error(f"Non-retryable error in LLM request: {e}")
                        raise
                else:
                    logger.error(f"LLM request failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    async def _generate_stream_with_retry(
        self, 
        request_params: Dict[str, Any]
    ) -> AsyncGenerator[LLMResult, None]:
        """Generate streaming text with retry logic"""
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                accumulated_content = ""
                
                async with self.session.post(
                    f"{self.base_url}/api/chat",
                    json=request_params
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode().strip())
                                    
                                    if "message" in data and "content" in data["message"]:
                                        content_chunk = data["message"]["content"]
                                        accumulated_content += content_chunk
                                        
                                        yield LLMResult(
                                            content=content_chunk,
                                            model=request_params["model"],
                                            input_tokens=None,  # Not available in streaming
                                            output_tokens=None,
                                            finish_reason=None,
                                            metadata={
                                                "is_partial": True,
                                                "accumulated_content": accumulated_content,
                                                "response_time_ms": (time.time() - start_time) * 1000
                                            }
                                        )
                                    
                                    # Check if done
                                    if data.get("done", False):
                                        self._request_count += 1
                                        
                                        # Final chunk with complete metadata
                                        yield LLMResult(
                                            content="",
                                            model=request_params["model"],
                                            input_tokens=data.get("prompt_eval_count"),
                                            output_tokens=data.get("eval_count"),
                                            finish_reason=data.get("done_reason", "stop"),
                                            metadata={
                                                "is_final": True,
                                                "accumulated_content": accumulated_content,
                                                "response_time_ms": (time.time() - start_time) * 1000,
                                                "eval_count": data.get("eval_count"),
                                                "eval_duration": data.get("eval_duration"),
                                                "load_duration": data.get("load_duration"),
                                                "prompt_eval_count": data.get("prompt_eval_count"),
                                                "prompt_eval_duration": data.get("prompt_eval_duration"),
                                                "total_duration": data.get("total_duration")
                                            }
                                        )
                                        
                                        # Update metrics
                                        input_tokens = data.get("prompt_eval_count", 0)
                                        output_tokens = data.get("eval_count", 0)
                                        self._total_input_tokens += input_tokens
                                        self._total_output_tokens += output_tokens
                                        
                                        return  # Successful completion
                                        
                                except json.JSONDecodeError:
                                    continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                
                return  # Successful completion without done=true (shouldn't happen)
                
            except Exception as e:
                self._error_count += 1
                
                if attempt < self.max_retries:
                    if self._is_retryable_error(e):
                        logger.warning(
                            f"Streaming LLM request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                            f"Retrying in {retry_delay:.1f}s"
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(
                            retry_delay * self.retry_exponential_base,
                            self.max_retry_delay
                        )
                        continue
                    else:
                        logger.error(f"Non-retryable error in streaming LLM request: {e}")
                        raise
                else:
                    logger.error(f"Streaming LLM request failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    async def _ensure_model_available(self, model: str) -> None:
        """Ensure the specified model is available, pull if necessary"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        try:
            # Check if model is already available
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    
                    # Check if model is available (exact match or with tag)
                    if any(model in m or m.startswith(model + ":") for m in available_models):
                        return
                    
                    logger.info(f"Model '{model}' not found locally. Available models: {available_models}")
                    
                    if self.auto_pull_models:
                        await self._pull_model(model)
                    else:
                        raise ValueError(
                            f"Model '{model}' not available and auto_pull_models is disabled. "
                            f"Available models: {available_models}"
                        )
                else:
                    raise Exception(f"Failed to check available models: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            raise
    
    async def _pull_model(self, model: str) -> None:
        """Pull a model from Ollama registry"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        logger.info(f"Pulling model '{model}' from Ollama registry...")
        start_time = time.time()
        
        try:
            payload = {"name": model}
            
            # Use a longer timeout for model pulling
            pull_timeout = ClientTimeout(total=self.pull_timeout)
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=pull_timeout
            ) as response:
                if response.status == 200:
                    # Stream the pull progress
                    async for line in response.content:
                        if line:
                            try:
                                progress_data = json.loads(line.decode().strip())
                                if "status" in progress_data:
                                    status = progress_data["status"]
                                    if "completed" in progress_data and "total" in progress_data:
                                        completed = progress_data["completed"]
                                        total = progress_data["total"]
                                        percent = (completed / total) * 100 if total > 0 else 0
                                        logger.info(f"Pulling {model}: {status} ({percent:.1f}%)")
                                    else:
                                        logger.info(f"Pulling {model}: {status}")
                            except json.JSONDecodeError:
                                continue
                    
                    pull_time = time.time() - start_time
                    self._model_pull_count += 1
                    logger.info(f"Successfully pulled model '{model}' in {pull_time:.1f}s")
                    
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to pull model: HTTP {response.status} - {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Model pull timed out after {self.pull_timeout}s")
        except Exception as e:
            logger.error(f"Failed to pull model '{model}': {e}")
            raise
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()
        
        # Connection errors
        if any(term in error_str for term in ["connection", "timeout", "network"]):
            return True
        
        # HTTP server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True
        
        # Ollama specific errors
        if "model not found" in error_str:
            return False  # Don't retry if model doesn't exist
        
        # Rate limiting (unlikely with local Ollama but possible)
        if "rate limit" in error_str or "429" in error_str:
            return True
        
        # aiohttp specific errors
        if isinstance(error, (ClientError, asyncio.TimeoutError)):
            return True
        
        return False
    
    @property
    def default_model(self) -> str:
        """Default LLM model"""
        return "llama3.1"
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported LLM models"""
        return list(self.MODEL_CONFIGS.keys())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics"""
        return {
            "request_count": self._request_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "error_count": self._error_count,
            "model_pull_count": self._model_pull_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "model": self.model
        }


class OllamaProvider(HybridAIProvider, RerankingProvider):
    """Combined Ollama provider for embeddings, LLM, and reranking"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama hybrid provider"""
        # Initialize sub-providers first
        self.embedding_provider = OllamaEmbeddingProvider(config)
        self.llm_provider = OllamaLLMProvider(config)
        
        # Shared session will be created in initialize()
        self.session: Optional[ClientSession] = None
        
        # Call parent constructor after sub-providers are created
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Validate configuration for both providers"""
        # The sub-providers will handle their own validation
        self.embedding_provider._validate_config()
        self.llm_provider._validate_config()
    
    async def initialize(self) -> None:
        """Initialize both providers with shared session"""
        try:
            # Create shared HTTP session
            base_url = self.config.get("base_url", "http://localhost:11434")
            timeout = self.config.get("timeout", 300.0)
            connect_timeout = self.config.get("connect_timeout", 10.0)
            
            session_timeout = ClientTimeout(
                total=timeout,
                connect=connect_timeout
            )
            
            self.session = ClientSession(
                timeout=session_timeout,
                connector=aiohttp.TCPConnector(limit=20, limit_per_host=10)
            )
            
            # Share session with sub-providers
            self.embedding_provider.session = self.session
            self.llm_provider.session = self.session
            
            # Initialize sub-providers (they'll skip session creation)
            await self.embedding_provider.initialize()
            await self.llm_provider.initialize()
            
            logger.info("Ollama hybrid provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama hybrid provider: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
    
    async def close(self) -> None:
        """Close both providers and shared session"""
        # Close sub-providers first
        if hasattr(self.embedding_provider, 'session'):
            self.embedding_provider.session = None  # Don't close twice
        if hasattr(self.llm_provider, 'session'):
            self.llm_provider.session = None  # Don't close twice
        
        await self.embedding_provider.close()
        await self.llm_provider.close()
        
        # Close shared session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Ollama hybrid provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Combined health check for both embedding and LLM capabilities"""
        embedding_health = await self.embedding_provider.health_check()
        llm_health = await self.llm_provider.health_check()
        
        # Return combined status
        is_healthy = embedding_health.is_healthy and llm_health.is_healthy
        avg_response_time = (embedding_health.response_time_ms + llm_health.response_time_ms) / 2
        
        error_messages = []
        if embedding_health.error_message:
            error_messages.append(f"Embedding: {embedding_health.error_message}")
        if llm_health.error_message:
            error_messages.append(f"LLM: {llm_health.error_message}")
        
        return HealthStatus(
            is_healthy=is_healthy,
            provider=AIProvider.OLLAMA.value,
            model=f"Embedding: {self.default_embedding_model}, LLM: {self.default_llm_model}",
            response_time_ms=avg_response_time,
            error_message="; ".join(error_messages) if error_messages else None,
            metadata={
                "embedding_health": embedding_health.__dict__,
                "llm_health": llm_health.__dict__,
                "base_url": self.config.get("base_url", "http://localhost:11434")
            }
        )
    
    # Embedding methods (delegate to embedding provider)
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
    
    # LLM methods (delegate to LLM provider)
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
    ) -> AsyncGenerator[LLMResult, None]:
        async for result in self.llm_provider.generate_stream(
            messages, model, temperature, max_tokens, **kwargs
        ):
            yield result
    
    # Properties
    @property
    def default_embedding_model(self) -> str:
        return self.embedding_provider.default_model
    
    @property
    def default_llm_model(self) -> str:
        return self.llm_provider.default_model
    
    @property
    def supported_models(self) -> List[str]:
        """All supported models (embedding + LLM)"""
        return self.embedding_provider.supported_models + self.llm_provider.supported_models
    
    @property
    def max_batch_size(self) -> int:
        return self.embedding_provider.max_batch_size
    
    @property
    def max_text_length(self) -> int:
        return self.embedding_provider.max_text_length
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from both providers"""
        embedding_metrics = self.embedding_provider.get_metrics()
        llm_metrics = self.llm_provider.get_metrics()
        
        return {
            "embedding": embedding_metrics,
            "llm": llm_metrics,
            "combined": {
                "total_requests": embedding_metrics["request_count"] + llm_metrics["request_count"],
                "total_errors": embedding_metrics["error_count"] + llm_metrics["error_count"],
                "total_model_pulls": embedding_metrics["model_pull_count"] + llm_metrics["model_pull_count"]
            }
        }
    
    async def list_available_models(self) -> Dict[str, List[str]]:
        """List models available in the local Ollama instance"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        try:
            base_url = self.config.get("base_url", "http://localhost:11434")
            async with self.session.get(f"{base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [model["name"] for model in data.get("models", [])]
                    
                    # Categorize models (best effort)
                    embedding_models = []
                    llm_models = []
                    
                    for model in available_models:
                        model_name = model.split(":")[0]  # Remove tag
                        if model_name in self.embedding_provider.MODEL_CONFIGS:
                            embedding_models.append(model)
                        elif model_name in self.llm_provider.MODEL_CONFIGS:
                            llm_models.append(model)
                        else:
                            # Unknown model, assume it's an LLM
                            llm_models.append(model)
                    
                    return {
                        "embedding_models": embedding_models,
                        "llm_models": llm_models,
                        "all_models": available_models
                    }
                else:
                    raise Exception(f"Failed to list models: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to list available models: {e}")
            raise
    
    async def pull_model(self, model: str) -> None:
        """Pull a specific model (can be used by either provider)"""
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        # Try to determine which provider should handle this model
        if model in self.embedding_provider.MODEL_CONFIGS:
            await self.embedding_provider._pull_model(model)
        elif model in self.llm_provider.MODEL_CONFIGS:
            await self.llm_provider._pull_model(model)
        elif model in self.embedding_provider.RERANKING_MODEL_CONFIGS:
            # Reranking models can be pulled using embedding provider's pull method
            await self.embedding_provider._pull_model(model)
        else:
            # Unknown model, try with LLM provider
            await self.llm_provider._pull_model(model)
    
    # Reranking methods (using Ollama reranking models)
    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> RerankingResult:
        """Rerank search results using Ollama reranking models
        
        Args:
            query: Search query for relevance scoring
            results: List of search results to rerank
            model: Optional model override for reranking
            
        Returns:
            RerankingResult with reranked results and scores
        """
        if not self.session:
            raise ConnectionError("Session not initialized")
        
        # Validate input
        self.validate_reranking_input(query, results)
        
        use_model = model or self.default_reranking_model
        
        # Ensure reranking model is available
        await self._ensure_reranking_model_available(use_model)
        
        start_time = time.time()
        
        try:
            # Prepare query-document pairs for reranking
            rerank_scores = []
            
            for result in results:
                content = result.get("content", "")
                if isinstance(content, str) and content.strip():
                    score = await self._rerank_single_pair(query, content, use_model)
                    rerank_scores.append(score)
                else:
                    # Handle empty content with zero score
                    rerank_scores.append(0.0)
            
            # Create reranked results with scores
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, rerank_scores)):
                enhanced_result = result.copy()
                enhanced_result["rerank_score"] = float(score)
                enhanced_result["original_rank"] = i
                reranked_results.append(enhanced_result)
            
            # Sort by rerank score in descending order
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Reranked {len(results)} results using Ollama model {use_model} "
                f"in {processing_time_ms:.1f}ms"
            )
            
            return RerankingResult(
                results=reranked_results,
                model=use_model,
                provider=AIProvider.OLLAMA.value,
                rerank_scores=rerank_scores,
                processing_time_ms=processing_time_ms,
                metadata={
                    "base_url": self.config.get("base_url", "http://localhost:11434"),
                    "reranking_method": "ollama_api",
                    "model_info": self.embedding_provider.RERANKING_MODEL_CONFIGS.get(
                        use_model, {}
                    ),
                    "num_pairs": len(results)
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama reranking failed: {e}")
            raise
    
    async def _ensure_reranking_model_available(self, model: str) -> None:
        """Ensure reranking model is available in Ollama"""
        try:
            # Check if model is already available
            async with self.session.get(f"{self.config.get('base_url', 'http://localhost:11434')}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["name"] for m in data.get("models", [])]
                    
                    # Check if model is available (exact match or with tag)
                    if any(model in m or m.startswith(model + ":") for m in available_models):
                        return
                    
                    logger.info(f"Reranking model '{model}' not found locally. Available models: {available_models}")
                    
                    # Pull the model if auto-pull is enabled
                    if self.embedding_provider.auto_pull_models:
                        await self.embedding_provider._pull_model(model)
                    else:
                        raise ValueError(
                            f"Reranking model '{model}' not available and auto_pull_models is disabled. "
                            f"Available models: {available_models}"
                        )
                else:
                    raise Exception(f"Failed to check available models: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to ensure reranking model availability: {e}")
            raise
    
    async def _rerank_single_pair(self, query: str, document: str, model: str) -> float:
        """Get reranking score for a single query-document pair
        
        Args:
            query: Search query
            document: Document content to score
            model: Reranking model to use
            
        Returns:
            Relevance score (higher means more relevant)
        """
        base_url = self.config.get("base_url", "http://localhost:11434")
        
        # Different reranking models may use different prompts
        if "bge-reranker" in model:
            # BGE reranker models expect a specific format
            prompt = f"Query: {query}\nDocument: {document}\nRelevance:"
        else:
            # Generic reranking prompt
            prompt = f"Determine the relevance of the following document to the query. Return a score between 0 and 1.\n\nQuery: {query}\n\nDocument: {document}\n\nRelevance score:"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,  # Deterministic scoring
                "num_predict": 10,   # Short response expected
                "stop": ["\n", "\n\n"]
            }
        }
        
        try:
            async with self.session.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=ClientTimeout(total=30.0)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "").strip()
                    
                    # Parse the score from the response
                    score = self._parse_reranking_score(response_text)
                    return score
                else:
                    error_text = await response.text()
                    logger.warning(f"Reranking request failed: HTTP {response.status} - {error_text}")
                    return 0.0
                    
        except asyncio.TimeoutError:
            logger.warning(f"Reranking request timed out for model {model}")
            return 0.0
        except Exception as e:
            logger.warning(f"Error in reranking request: {e}")
            return 0.0
    
    def _parse_reranking_score(self, response_text: str) -> float:
        """Parse reranking score from model response
        
        Args:
            response_text: Raw response from reranking model
            
        Returns:
            Normalized score between 0.0 and 1.0
        """
        try:
            # Try to extract a numeric score
            import re
            
            # Look for decimal numbers in the response
            numbers = re.findall(r'\d*\.?\d+', response_text)
            
            if numbers:
                score = float(numbers[0])
                
                # Normalize score to 0-1 range
                if score > 1.0:
                    # If score is > 1, assume it's on a different scale (like 0-10)
                    if score <= 10:
                        score = score / 10.0
                    elif score <= 100:
                        score = score / 100.0
                    else:
                        score = min(score / 1000.0, 1.0)
                
                return max(0.0, min(1.0, score))
            
            # If no numbers found, try to interpret text
            response_lower = response_text.lower()
            if any(word in response_lower for word in ["high", "very relevant", "excellent"]):
                return 0.9
            elif any(word in response_lower for word in ["good", "relevant", "moderate"]):
                return 0.7
            elif any(word in response_lower for word in ["low", "poor", "irrelevant"]):
                return 0.3
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.warning(f"Error parsing reranking score from '{response_text}': {e}")
            return 0.0
    
    def supports_reranking(self) -> bool:
        """Check if provider supports reranking functionality"""
        use_reranking = self.config.get("use_reranking", False)
        reranking_provider = self.config.get("reranking_provider", "")
        return use_reranking and reranking_provider == "ollama"
    
    def get_reranking_models(self) -> List[str]:
        """Get list of available reranking models"""
        return list(self.embedding_provider.RERANKING_MODEL_CONFIGS.keys())
    
    @property
    def default_reranking_model(self) -> str:
        """Default reranking model for Ollama provider"""
        return "bge-reranker-base"
    
    @property
    def max_results_count(self) -> int:
        """Maximum number of results that can be reranked at once"""
        # Ollama reranking is done one pair at a time, so we can handle more results
        # but we need to be mindful of processing time
        return self.config.get("reranking_max_results", 50)