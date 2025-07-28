"""OpenAI AI provider implementation"""

import asyncio
import logging
import os
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI
from openai.types import Embedding, CreateEmbeddingResponse

from ..base import (
    EmbeddingProvider, 
    LLMProvider, 
    HybridAIProvider,
    EmbeddingResult, 
    LLMResult, 
    HealthStatus,
    AIProvider
)

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation"""
    
    # Model configurations with dimensions and context lengths
    MODEL_CONFIGS = {
        "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI embedding provider
        
        Args:
            config: Configuration dictionary with OpenAI settings
        """
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        self.model = config.get("embedding_model", self.default_model)
        
        # Rate limiting and retry settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 1.0)
        self.max_retry_delay = config.get("max_retry_delay", 60.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)
        
        # Batch processing settings
        self._batch_size = config.get("batch_size", 50)  # OpenAI supports up to 2048
        
        # Performance monitoring
        self._request_count = 0
        self._total_tokens = 0
        self._error_count = 0
        
    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration"""
        api_key = self.config.get("api_key")
        if not api_key:
            # Try environment variable as fallback
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key must be provided in config['api_key'] or "
                    "OPENAI_API_KEY environment variable"
                )
            self.config["api_key"] = api_key
        
        # Validate model if specified
        model = self.config.get("embedding_model")
        if model and model not in self.MODEL_CONFIGS:
            logger.warning(
                f"Model '{model}' not in known configurations. "
                f"Supported models: {list(self.MODEL_CONFIGS.keys())}"
            )
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                timeout=self.config.get("timeout", 60.0),
                max_retries=0  # We handle retries manually for better control
            )
            
            # Test connection with a small embedding
            await self.health_check()
            logger.info("OpenAI embedding provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embedding provider: {e}")
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")
    
    async def close(self) -> None:
        """Close the OpenAI client"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("OpenAI embedding provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Check if OpenAI embeddings API is healthy"""
        if not self.client:
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OPENAI.value,
                model=self.model,
                response_time_ms=0.0,
                error_message="Client not initialized"
            )
        
        start_time = time.time()
        try:
            # Test with a simple embedding request
            response = await self.client.embeddings.create(
                model=self.model,
                input=["test"],
                timeout=10.0
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.data and len(response.data) > 0:
                return HealthStatus(
                    is_healthy=True,
                    provider=AIProvider.OPENAI.value,
                    model=self.model,
                    response_time_ms=response_time_ms,
                    metadata={
                        "model": response.model,
                        "usage": response.usage.dict() if response.usage else None
                    }
                )
            else:
                return HealthStatus(
                    is_healthy=False,
                    provider=AIProvider.OPENAI.value,
                    model=self.model,
                    response_time_ms=response_time_ms,
                    error_message="Empty response from API"
                )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OPENAI.value,
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
        """Create embeddings for multiple texts in batches"""
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
        
        # Process in batches
        all_results = []
        
        for i in range(0, len(valid_texts), use_batch_size):
            batch_end = min(i + use_batch_size, len(valid_texts))
            batch_texts = valid_texts[i:batch_end]
            
            try:
                batch_results = await self._create_embeddings_batch_with_retry(
                    batch_texts, use_model
                )
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//use_batch_size + 1}: {e}")
                
                # Try individual fallback for this batch
                logger.info("Attempting individual embedding creation as fallback")
                for text in batch_texts:
                    try:
                        individual_result = await self._create_embeddings_batch_with_retry(
                            [text], use_model
                        )
                        all_results.extend(individual_result)
                    except Exception as individual_error:
                        logger.error(f"Individual embedding also failed: {individual_error}")
                        # Create zero embedding as last resort
                        dimensions = self.get_model_dimensions(use_model)
                        all_results.append(EmbeddingResult(
                            embedding=[0.0] * dimensions,
                            dimensions=dimensions,
                            model=use_model,
                            input_tokens=0
                        ))
        
        # Handle empty text indices by inserting zero embeddings
        for empty_idx in empty_indices:
            dimensions = self.get_model_dimensions(use_model)
            zero_result = EmbeddingResult(
                embedding=[0.0] * dimensions,
                dimensions=dimensions,
                model=use_model,
                input_tokens=0
            )
            all_results.insert(empty_idx, zero_result)
        
        return all_results
    
    async def _create_embeddings_batch_with_retry(
        self, 
        texts: List[str], 
        model: str
    ) -> List[EmbeddingResult]:
        """Create embeddings with exponential backoff retry logic"""
        if not self.client:
            raise ConnectionError("Client not initialized")
        
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response: CreateEmbeddingResponse = await self.client.embeddings.create(
                    model=model,
                    input=texts
                )
                
                # Update metrics
                self._request_count += 1
                if response.usage:
                    self._total_tokens += response.usage.total_tokens
                
                # Convert to EmbeddingResult objects
                results = []
                dimensions = self.get_model_dimensions(model)
                
                for i, embedding_data in enumerate(response.data):
                    results.append(EmbeddingResult(
                        embedding=embedding_data.embedding,
                        dimensions=dimensions,
                        model=response.model,
                        input_tokens=response.usage.total_tokens // len(texts) if response.usage else None
                    ))
                
                processing_time = (time.time() - start_time) * 1000
                logger.debug(
                    f"Created {len(results)} embeddings in {processing_time:.1f}ms "
                    f"(attempt {attempt + 1})"
                )
                
                return results
                
            except Exception as e:
                self._error_count += 1
                
                if attempt < self.max_retries:
                    # Check if it's a retryable error
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
                        # Non-retryable error, fail immediately
                        logger.error(f"Non-retryable error in embedding request: {e}")
                        raise
                else:
                    # Final attempt failed
                    logger.error(f"Embedding request failed after {self.max_retries + 1} attempts: {e}")
                    raise
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()
        
        # Rate limit errors
        if "rate limit" in error_str or "429" in error_str:
            return True
        
        # Temporary server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True
        
        # Network/connection errors
        if any(term in error_str for term in ["timeout", "connection", "network"]):
            return True
        
        # OpenAI specific retryable errors
        if isinstance(error, openai.RateLimitError):
            return True
        if isinstance(error, openai.InternalServerError):
            return True
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIConnectionError):
            return True
        
        return False
    
    @property
    def default_model(self) -> str:
        """Default embedding model"""
        return "text-embedding-3-small"
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported embedding models"""
        return list(self.MODEL_CONFIGS.keys())
    
    def get_model_dimensions(self, model: str) -> int:
        """Get embedding dimensions for a model"""
        if model in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model]["dimensions"]
        
        # Default for unknown models
        logger.warning(f"Unknown model '{model}', using default dimensions")
        return 1536
    
    @property
    def max_batch_size(self) -> int:
        """Maximum batch size for OpenAI embeddings"""
        return self._batch_size
    
    @property
    def max_text_length(self) -> int:
        """Maximum text length for OpenAI embeddings"""
        model_config = self.MODEL_CONFIGS.get(self.model, {})
        return model_config.get("max_tokens", 8191) * 4  # Rough token to char conversion
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get provider performance metrics"""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "model": self.model
        }


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider implementation"""
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4": {"max_tokens": 8192, "context_length": 8192},
        "gpt-4-32k": {"max_tokens": 32768, "context_length": 32768},
        "gpt-4-turbo": {"max_tokens": 4096, "context_length": 128000},
        "gpt-4-turbo-preview": {"max_tokens": 4096, "context_length": 128000},
        "gpt-4o": {"max_tokens": 4096, "context_length": 128000},
        "gpt-4o-mini": {"max_tokens": 16384, "context_length": 128000},
        "gpt-3.5-turbo": {"max_tokens": 4096, "context_length": 16385},
        "gpt-3.5-turbo-16k": {"max_tokens": 16384, "context_length": 16385},
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI LLM provider"""
        super().__init__(config)
        self.client: Optional[AsyncOpenAI] = None
        self.model = config.get("llm_model", self.default_model)
        
        # Generation settings
        self._temperature = config.get("temperature", self.default_temperature)
        self._max_tokens = config.get("max_tokens", self.default_max_tokens)
        
        # Rate limiting and retry settings
        self.max_retries = config.get("max_retries", 3)
        self.initial_retry_delay = config.get("initial_retry_delay", 1.0)
        self.max_retry_delay = config.get("max_retry_delay", 60.0)
        self.retry_exponential_base = config.get("retry_exponential_base", 2.0)
        
        # Performance monitoring
        self._request_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._error_count = 0
    
    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration"""
        api_key = self.config.get("api_key")
        if not api_key:
            # Try environment variable as fallback
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key must be provided in config['api_key'] or "
                    "OPENAI_API_KEY environment variable"
                )
            self.config["api_key"] = api_key
        
        # Validate model if specified
        model = self.config.get("llm_model")
        if model and model not in self.MODEL_CONFIGS:
            logger.warning(
                f"Model '{model}' not in known configurations. "
                f"Supported models: {list(self.MODEL_CONFIGS.keys())}"
            )
    
    async def initialize(self) -> None:
        """Initialize the OpenAI client"""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                timeout=self.config.get("timeout", 60.0),
                max_retries=0  # We handle retries manually
            )
            
            # Test connection
            await self.health_check()
            logger.info("OpenAI LLM provider initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM provider: {e}")
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}")
    
    async def close(self) -> None:
        """Close the OpenAI client"""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("OpenAI LLM provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Check if OpenAI chat API is healthy"""
        if not self.client:
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OPENAI.value,
                model=self.model,
                response_time_ms=0.0,
                error_message="Client not initialized"
            )
        
        start_time = time.time()
        try:
            # Test with a simple chat completion
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=10.0
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            if response.choices and len(response.choices) > 0:
                return HealthStatus(
                    is_healthy=True,
                    provider=AIProvider.OPENAI.value,
                    model=self.model,
                    response_time_ms=response_time_ms,
                    metadata={
                        "model": response.model,
                        "usage": response.usage.dict() if response.usage else None
                    }
                )
            else:
                return HealthStatus(
                    is_healthy=False,
                    provider=AIProvider.OPENAI.value,
                    model=self.model,
                    response_time_ms=response_time_ms,
                    error_message="Empty response from API"
                )
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            return HealthStatus(
                is_healthy=False,
                provider=AIProvider.OPENAI.value,
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
        """Generate text using OpenAI chat completion"""
        if not self.client:
            raise ConnectionError("Client not initialized")
        
        # Validate messages
        self.validate_messages(messages)
        
        use_model = model or self.model
        use_temperature = temperature if temperature is not None else self._temperature
        use_max_tokens = max_tokens or self._max_tokens
        
        # Prepare request parameters
        request_params = {
            "model": use_model,
            "messages": messages,
            "temperature": use_temperature,
            "max_tokens": use_max_tokens,
            **kwargs
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
        if not self.client:
            raise ConnectionError("Client not initialized")
        
        # Validate messages
        self.validate_messages(messages)
        
        use_model = model or self.model
        use_temperature = temperature if temperature is not None else self._temperature
        use_max_tokens = max_tokens or self._max_tokens
        
        # Prepare request parameters
        request_params = {
            "model": use_model,
            "messages": messages,
            "temperature": use_temperature,
            "max_tokens": use_max_tokens,
            "stream": True,
            **kwargs
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
                
                response = await self.client.chat.completions.create(**request_params)
                
                # Update metrics
                self._request_count += 1
                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens
                    self._total_output_tokens += response.usage.completion_tokens
                
                # Extract result
                choice = response.choices[0]
                content = choice.message.content or ""
                
                result = LLMResult(
                    content=content,
                    model=response.model,
                    input_tokens=response.usage.prompt_tokens if response.usage else None,
                    output_tokens=response.usage.completion_tokens if response.usage else None,
                    finish_reason=choice.finish_reason,
                    metadata={
                        "usage": response.usage.dict() if response.usage else None,
                        "response_time_ms": (time.time() - start_time) * 1000
                    }
                )
                
                logger.debug(
                    f"Generated {len(content)} characters in "
                    f"{result.metadata['response_time_ms']:.1f}ms (attempt {attempt + 1})"
                )
                
                return result
                
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
                
                stream = await self.client.chat.completions.create(**request_params)
                
                async for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        choice = chunk.choices[0]
                        
                        if choice.delta and choice.delta.content:
                            content_chunk = choice.delta.content
                            accumulated_content += content_chunk
                            
                            yield LLMResult(
                                content=content_chunk,
                                model=chunk.model or request_params["model"],
                                input_tokens=None,  # Not available in streaming
                                output_tokens=None,
                                finish_reason=choice.finish_reason,
                                metadata={
                                    "is_partial": True,
                                    "accumulated_content": accumulated_content,
                                    "response_time_ms": (time.time() - start_time) * 1000
                                }
                            )
                        
                        # Final chunk
                        if choice.finish_reason:
                            self._request_count += 1
                            yield LLMResult(
                                content="",
                                model=chunk.model or request_params["model"],
                                input_tokens=None,
                                output_tokens=None,
                                finish_reason=choice.finish_reason,
                                metadata={
                                    "is_final": True,
                                    "accumulated_content": accumulated_content,
                                    "response_time_ms": (time.time() - start_time) * 1000
                                }
                            )
                
                return  # Successful completion
                
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
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable"""
        error_str = str(error).lower()
        
        # Rate limit errors
        if "rate limit" in error_str or "429" in error_str:
            return True
        
        # Temporary server errors
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True
        
        # Network/connection errors
        if any(term in error_str for term in ["timeout", "connection", "network"]):
            return True
        
        # OpenAI specific retryable errors
        if isinstance(error, openai.RateLimitError):
            return True
        if isinstance(error, openai.InternalServerError):
            return True
        if isinstance(error, openai.APITimeoutError):
            return True
        if isinstance(error, openai.APIConnectionError):
            return True
        
        return False
    
    @property
    def default_model(self) -> str:
        """Default LLM model"""
        return "gpt-4o-mini"
    
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
            "error_rate": self._error_count / max(self._request_count, 1),
            "model": self.model
        }


class OpenAIProvider(HybridAIProvider):
    """Combined OpenAI provider for both embeddings and LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI hybrid provider"""
        super().__init__(config)
        
        # Initialize sub-providers
        self.embedding_provider = OpenAIEmbeddingProvider(config)
        self.llm_provider = OpenAILLMProvider(config)
        
        self.client: Optional[AsyncOpenAI] = None
    
    def _validate_config(self) -> None:
        """Validate configuration for both providers"""
        # The sub-providers will handle their own validation
        self.embedding_provider._validate_config()
        self.llm_provider._validate_config()
    
    async def initialize(self) -> None:
        """Initialize both providers"""
        # Create shared client
        self.client = AsyncOpenAI(
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 60.0),
            max_retries=0
        )
        
        # Share client with sub-providers
        self.embedding_provider.client = self.client
        self.llm_provider.client = self.client
        
        # Initialize sub-providers (they'll skip client creation)
        await self.embedding_provider.initialize()
        await self.llm_provider.initialize()
        
        logger.info("OpenAI hybrid provider initialized successfully")
    
    async def close(self) -> None:
        """Close both providers"""
        await self.embedding_provider.close()
        await self.llm_provider.close()
        
        if self.client:
            await self.client.close()
            self.client = None
        
        logger.info("OpenAI hybrid provider closed")
    
    async def health_check(self) -> HealthStatus:
        """Combined health check"""
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
            provider=AIProvider.OPENAI.value,
            model=f"Embedding: {self.default_embedding_model}, LLM: {self.default_llm_model}",
            response_time_ms=avg_response_time,
            error_message="; ".join(error_messages) if error_messages else None,
            metadata={
                "embedding_health": embedding_health.__dict__,
                "llm_health": llm_health.__dict__
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
        """Get combined metrics"""
        embedding_metrics = self.embedding_provider.get_metrics()
        llm_metrics = self.llm_provider.get_metrics()
        
        return {
            "embedding": embedding_metrics,
            "llm": llm_metrics,
            "combined": {
                "total_requests": embedding_metrics["request_count"] + llm_metrics["request_count"],
                "total_errors": embedding_metrics["error_count"] + llm_metrics["error_count"]
            }
        }