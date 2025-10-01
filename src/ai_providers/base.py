"""Abstract base classes for AI providers"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    dimensions: int
    model: str
    input_tokens: Optional[int] = None
    
    def __post_init__(self):
        """Validate embedding dimensions"""
        if len(self.embedding) != self.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimensions}, "
                f"got {len(self.embedding)}"
            )


@dataclass
class LLMResult:
    """Result from LLM generation"""
    content: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HealthStatus:
    """Health check status for AI providers"""
    is_healthy: bool
    provider: str
    model: str
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RerankingResult:
    """Result from reranking operation"""
    results: List[Dict[str, Any]]  # Reranked results with enhanced scores
    model: str                     # Model used for reranking
    provider: str                  # Provider that performed reranking
    rerank_scores: List[float]     # Individual reranking scores
    processing_time_ms: float      # Performance metrics
    metadata: Optional[Dict[str, Any]] = None  # Additional provider-specific info


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider with configuration
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection and resources"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection and clean up resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the provider is healthy and responsive
        
        Returns:
            HealthStatus with provider health information
        """
        pass
    
    @abstractmethod
    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResult:
        """Create an embedding for a single text
        
        Args:
            text: Text to create embedding for
            model: Optional model override
            
        Returns:
            EmbeddingResult with the generated embedding
            
        Raises:
            ValueError: If text is empty or too long
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass
    
    @abstractmethod
    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> List[EmbeddingResult]:
        """Create embeddings for multiple texts in batches
        
        Args:
            texts: List of texts to create embeddings for
            model: Optional model override
            batch_size: Optional batch size override
            
        Returns:
            List of EmbeddingResult objects
            
        Raises:
            ValueError: If texts list is empty or contains invalid text
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default embedding model for this provider"""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported embedding models"""
        pass
    
    @abstractmethod
    def get_model_dimensions(self, model: str) -> int:
        """Get the embedding dimensions for a specific model
        
        Args:
            model: Model name
            
        Returns:
            Number of dimensions for the model
            
        Raises:
            ValueError: If model is not supported
        """
        pass
    
    @property
    def max_batch_size(self) -> int:
        """Maximum batch size for embedding generation"""
        return 100  # Default safe batch size
    
    @property
    def max_text_length(self) -> int:
        """Maximum text length for embedding generation"""
        return 8192  # Default safe text length


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider with configuration
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection and resources"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection and clean up resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the provider is healthy and responsive
        
        Returns:
            HealthStatus with provider health information
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResult:
        """Generate text using the LLM
        
        Args:
            messages: List of messages in chat format [{"role": "user", "content": "..."}]
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResult with the generated content
            
        Raises:
            ValueError: If messages format is invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Generate text using the LLM with streaming response
        
        Args:
            messages: List of messages in chat format
            model: Optional model override
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Partial LLMResult objects as content is generated
            
        Raises:
            ValueError: If messages format is invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default LLM model for this provider"""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported LLM models"""
        pass
    
    @property
    def default_temperature(self) -> float:
        """Default temperature for text generation"""
        return 0.7
    
    @property
    def default_max_tokens(self) -> int:
        """Default max tokens for text generation"""
        return 1000
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format
        
        Args:
            messages: List of messages to validate
            
        Raises:
            ValueError: If messages format is invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            
            if "role" not in message:
                raise ValueError(f"Message {i} missing 'role' field")
            
            if "content" not in message:
                raise ValueError(f"Message {i} missing 'content' field")
            
            if message["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"Message {i} has invalid role '{message['role']}'. "
                    "Must be 'system', 'user', or 'assistant'"
                )
            
            if not isinstance(message["content"], str):
                raise ValueError(f"Message {i} content must be a string")


class RerankingProvider(ABC):
    """Abstract base class for reranking providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the reranking provider with configuration
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection and resources"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection and clean up resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the provider is healthy and responsive
        
        Returns:
            HealthStatus with provider health information
        """
        pass
    
    @abstractmethod
    async def rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> RerankingResult:
        """Rerank search results based on query relevance
        
        Args:
            query: Search query for relevance scoring
            results: List of search results to rerank
            model: Optional model override for reranking
            
        Returns:
            RerankingResult with reranked results and scores
            
        Raises:
            ValueError: If query is empty or results invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass
    
    @abstractmethod
    def supports_reranking(self) -> bool:
        """Check if provider supports reranking functionality
        
        Returns:
            True if reranking is supported and available
        """
        pass
    
    @abstractmethod
    def get_reranking_models(self) -> List[str]:
        """Get list of available reranking models
        
        Returns:
            List of model names available for reranking
        """
        pass
    
    @property
    @abstractmethod
    def default_reranking_model(self) -> str:
        """Default reranking model for this provider"""
        pass
    
    @property
    def max_results_count(self) -> int:
        """Maximum number of results that can be reranked at once"""
        return 100  # Default safe limit
    
    def validate_reranking_input(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Validate reranking input parameters
        
        Args:
            query: Search query to validate
            results: Results list to validate
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        if not results:
            raise ValueError("Results list cannot be empty")
        
        if len(results) > self.max_results_count:
            raise ValueError(
                f"Too many results to rerank: {len(results)}. "
                f"Maximum allowed: {self.max_results_count}"
            )
        
        # Validate results format
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                raise ValueError(f"Result {i} must be a dictionary")
            
            if "content" not in result:
                raise ValueError(f"Result {i} missing required 'content' field")
            
            if not isinstance(result["content"], str):
                raise ValueError(f"Result {i} content must be a string")


class HybridAIProvider(EmbeddingProvider, LLMProvider):
    """Abstract base class for providers that support both embeddings and LLM"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the hybrid provider with configuration
        
        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()
    
    @property
    @abstractmethod 
    def default_embedding_model(self) -> str:
        """Default embedding model for this provider"""
        pass
    
    @property
    @abstractmethod
    def default_llm_model(self) -> str:
        """Default LLM model for this provider"""
        pass
    
    # Implement the abstract properties as aliases for backwards compatibility
    @property
    def default_model(self) -> str:
        """Default model (returns LLM model for backwards compatibility)"""
        return self.default_llm_model


@dataclass
class VisionResult:
    """Result from vision model operations"""
    content: str  # Caption, description, or extracted text
    model: str
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class VisionProvider(ABC):
    """Abstract base class for vision model providers"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vision provider with configuration

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider connection and resources"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection and clean up resources"""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the provider is healthy and responsive

        Returns:
            HealthStatus with provider health information
        """
        pass

    @abstractmethod
    async def generate_image_embedding(
        self,
        image_data: bytes,
        model: Optional[str] = None
    ) -> EmbeddingResult:
        """Generate vector embedding for an image

        Args:
            image_data: Image data as bytes
            model: Optional model override for vision embedding

        Returns:
            EmbeddingResult with the generated image embedding

        Raises:
            ValueError: If image data is invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass

    @abstractmethod
    async def generate_image_caption(
        self,
        image_data: bytes,
        prompt_template: Optional[str] = None,
        model: Optional[str] = None
    ) -> VisionResult:
        """Generate natural language description of an image

        Args:
            image_data: Image data as bytes
            prompt_template: Optional custom prompt template
            model: Optional model override for vision caption

        Returns:
            VisionResult with the generated caption

        Raises:
            ValueError: If image data is invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass

    @abstractmethod
    async def extract_text_from_image(
        self,
        image_data: bytes,
        model: Optional[str] = None
    ) -> VisionResult:
        """Extract text from an image using OCR

        Args:
            image_data: Image data as bytes
            model: Optional model override for OCR

        Returns:
            VisionResult with extracted text

        Raises:
            ValueError: If image data is invalid
            ConnectionError: If provider is unavailable
            Exception: For other provider-specific errors
        """
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if provider supports vision capabilities

        Returns:
            True if vision is supported and available
        """
        pass

    @property
    @abstractmethod
    def default_vision_model(self) -> str:
        """Default vision model for this provider"""
        pass

    @property
    @abstractmethod
    def supported_vision_models(self) -> List[str]:
        """List of supported vision models"""
        pass

    @property
    def max_image_size_mb(self) -> float:
        """Maximum image size in megabytes"""
        return 20.0  # Default 20MB limit

    @property
    def supported_image_formats(self) -> List[str]:
        """List of supported image formats"""
        return ["jpeg", "jpg", "png", "webp", "gif"]  # Common formats

    def validate_image_data(self, image_data: bytes) -> None:
        """Validate image data

        Args:
            image_data: Image data to validate

        Raises:
            ValueError: If image data is invalid
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")

        # Check size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > self.max_image_size_mb:
            raise ValueError(
                f"Image size {size_mb:.2f}MB exceeds maximum {self.max_image_size_mb}MB"
            )