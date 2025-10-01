"""Factory for creating AI provider instances"""

import logging
from typing import Dict, Any, List, Type, Union, Optional
from .base import AIProvider, EmbeddingProvider, LLMProvider, HybridAIProvider, RerankingProvider

logger = logging.getLogger(__name__)


class AIProviderFactory:
    """Factory for creating AI provider instances"""
    
    _embedding_providers: Dict[AIProvider, Type[EmbeddingProvider]] = {}
    _llm_providers: Dict[AIProvider, Type[LLMProvider]] = {}
    _hybrid_providers: Dict[AIProvider, Type[HybridAIProvider]] = {}
    _reranking_providers: Dict[AIProvider, Type[RerankingProvider]] = {}
    
    @classmethod
    def register_embedding_provider(
        cls, 
        provider_type: AIProvider, 
        provider_class: Type[EmbeddingProvider]
    ) -> None:
        """Register an embedding provider class for a given type
        
        Args:
            provider_type: AI provider enum value
            provider_class: Provider class that implements EmbeddingProvider
        """
        if not issubclass(provider_class, EmbeddingProvider):
            raise ValueError(
                f"Provider class must implement EmbeddingProvider interface"
            )
        
        cls._embedding_providers[provider_type] = provider_class
        logger.info(f"Registered embedding provider: {provider_type.value}")
    
    @classmethod
    def register_llm_provider(
        cls, 
        provider_type: AIProvider, 
        provider_class: Type[LLMProvider]
    ) -> None:
        """Register an LLM provider class for a given type
        
        Args:
            provider_type: AI provider enum value
            provider_class: Provider class that implements LLMProvider
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(
                f"Provider class must implement LLMProvider interface"
            )
        
        cls._llm_providers[provider_type] = provider_class
        logger.info(f"Registered LLM provider: {provider_type.value}")
    
    @classmethod
    def register_hybrid_provider(
        cls, 
        provider_type: AIProvider, 
        provider_class: Type[HybridAIProvider]
    ) -> None:
        """Register a hybrid provider class for a given type
        
        Args:
            provider_type: AI provider enum value
            provider_class: Provider class that implements HybridAIProvider
        """
        if not issubclass(provider_class, HybridAIProvider):
            raise ValueError(
                f"Provider class must implement HybridAIProvider interface"
            )
        
        cls._hybrid_providers[provider_type] = provider_class
        # Also register as both embedding and LLM provider
        cls._embedding_providers[provider_type] = provider_class
        cls._llm_providers[provider_type] = provider_class
        logger.info(f"Registered hybrid provider: {provider_type.value}")
    
    @classmethod
    def register_reranking_provider(
        cls,
        provider_type: AIProvider,
        provider_class: Type[RerankingProvider]
    ) -> None:
        """Register a reranking provider class for a given type
        
        Args:
            provider_type: AI provider enum value
            provider_class: Provider class that implements RerankingProvider
        """
        if not issubclass(provider_class, RerankingProvider):
            raise ValueError(
                f"Provider class must implement RerankingProvider interface"
            )
        
        cls._reranking_providers[provider_type] = provider_class
        logger.info(f"Registered reranking provider: {provider_type.value}")
    
    @classmethod
    def create_embedding_provider(
        cls, 
        provider_type: AIProvider, 
        config: Dict[str, Any]
    ) -> EmbeddingProvider:
        """Create an embedding provider instance
        
        Args:
            provider_type: AI provider enum value
            config: Provider configuration dictionary
            
        Returns:
            EmbeddingProvider instance
            
        Raises:
            ValueError: If provider type is not supported for embeddings
        """
        if provider_type not in cls._embedding_providers:
            available_providers = list(cls._embedding_providers.keys())
            raise ValueError(
                f"Unsupported embedding provider: {provider_type.value}. "
                f"Available: {[p.value for p in available_providers]}"
            )
        
        provider_class = cls._embedding_providers[provider_type]
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create embedding provider {provider_type.value}: {e}")
            raise
    
    @classmethod
    def create_llm_provider(
        cls, 
        provider_type: AIProvider, 
        config: Dict[str, Any]
    ) -> LLMProvider:
        """Create an LLM provider instance
        
        Args:
            provider_type: AI provider enum value
            config: Provider configuration dictionary
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider type is not supported for LLM
        """
        if provider_type not in cls._llm_providers:
            available_providers = list(cls._llm_providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: {provider_type.value}. "
                f"Available: {[p.value for p in available_providers]}"
            )
        
        provider_class = cls._llm_providers[provider_type]
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create LLM provider {provider_type.value}: {e}")
            raise
    
    @classmethod
    def create_hybrid_provider(
        cls, 
        provider_type: AIProvider, 
        config: Dict[str, Any]
    ) -> HybridAIProvider:
        """Create a hybrid provider instance
        
        Args:
            provider_type: AI provider enum value
            config: Provider configuration dictionary
            
        Returns:
            HybridAIProvider instance
            
        Raises:
            ValueError: If provider type is not supported as hybrid
        """
        if provider_type not in cls._hybrid_providers:
            available_providers = list(cls._hybrid_providers.keys())
            raise ValueError(
                f"Unsupported hybrid provider: {provider_type.value}. "
                f"Available: {[p.value for p in available_providers]}"
            )
        
        provider_class = cls._hybrid_providers[provider_type]
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create hybrid provider {provider_type.value}: {e}")
            raise
    
    @classmethod
    def create_reranking_provider(
        cls,
        provider_type: AIProvider,
        config: Dict[str, Any]
    ) -> RerankingProvider:
        """Create a reranking provider instance
        
        Args:
            provider_type: AI provider enum value
            config: Provider configuration dictionary
            
        Returns:
            RerankingProvider instance
            
        Raises:
            ValueError: If provider type is not supported for reranking
        """
        if provider_type not in cls._reranking_providers:
            available_providers = list(cls._reranking_providers.keys())
            raise ValueError(
                f"Unsupported reranking provider: {provider_type.value}. "
                f"Available: {[p.value for p in available_providers]}"
            )
        
        provider_class = cls._reranking_providers[provider_type]
        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create reranking provider {provider_type.value}: {e}")
            raise
    
    @classmethod
    def create_provider(
        cls,
        provider_type: AIProvider,
        config: Dict[str, Any],
        prefer_hybrid: bool = True
    ) -> Union[EmbeddingProvider, LLMProvider, HybridAIProvider]:
        """Create the best available provider instance
        
        Args:
            provider_type: AI provider enum value
            config: Provider configuration dictionary
            prefer_hybrid: Whether to prefer hybrid providers if available
            
        Returns:
            Provider instance (hybrid if available and preferred)
            
        Raises:
            ValueError: If provider type is not supported
        """
        # Try hybrid first if preferred and available
        if prefer_hybrid and provider_type in cls._hybrid_providers:
            return cls.create_hybrid_provider(provider_type, config)
        
        # If embedding is available, return embedding provider
        if provider_type in cls._embedding_providers:
            return cls.create_embedding_provider(provider_type, config)
        
        # Fall back to LLM provider
        if provider_type in cls._llm_providers:
            return cls.create_llm_provider(provider_type, config)
        
        # No providers available
        raise ValueError(f"No providers available for {provider_type.value}")
    
    @classmethod
    def create_provider_with_reranking(
        cls,
        provider_type: AIProvider,
        config: Dict[str, Any],
        reranking_provider_type: Optional[AIProvider] = None
    ) -> Union[EmbeddingProvider, LLMProvider, HybridAIProvider]:
        """Create a provider instance with optional reranking capability
        
        Args:
            provider_type: Primary AI provider enum value
            config: Provider configuration dictionary
            reranking_provider_type: Optional separate reranking provider type
            
        Returns:
            Provider instance with reranking capability if configured
        """
        # Create the primary provider
        primary_provider = cls.create_provider(provider_type, config)
        
        # Check if reranking is enabled and configured
        use_reranking = config.get("use_reranking", False)
        if not use_reranking:
            return primary_provider
        
        # Determine reranking provider
        reranking_provider_name = config.get("reranking_provider", "")
        if reranking_provider_type:
            actual_reranking_provider = reranking_provider_type
        elif reranking_provider_name:
            try:
                actual_reranking_provider = AIProvider(reranking_provider_name.lower())
            except ValueError:
                logger.warning(f"Invalid reranking provider: {reranking_provider_name}")
                return primary_provider
        else:
            return primary_provider
        
        # If the primary provider already supports reranking and it's the same provider, use it
        if (isinstance(primary_provider, RerankingProvider) and 
            provider_type == actual_reranking_provider):
            logger.info(f"Using {provider_type.value} provider for both primary and reranking")
            return primary_provider
        
        # If it's a different provider or primary doesn't support reranking,
        # we would need to create a composite provider, but for now return primary
        logger.info(f"Reranking provider {actual_reranking_provider.value} configured separately")
        return primary_provider
    
    @classmethod
    def get_available_embedding_providers(cls) -> List[AIProvider]:
        """Get list of available embedding providers"""
        return list(cls._embedding_providers.keys())
    
    @classmethod
    def get_available_llm_providers(cls) -> List[AIProvider]:
        """Get list of available LLM providers"""
        return list(cls._llm_providers.keys())
    
    @classmethod
    def get_available_hybrid_providers(cls) -> List[AIProvider]:
        """Get list of available hybrid providers"""
        return list(cls._hybrid_providers.keys())
    
    @classmethod
    def get_available_reranking_providers(cls) -> List[AIProvider]:
        """Get list of available reranking providers"""
        return list(cls._reranking_providers.keys())
    
    @classmethod
    def get_provider_capabilities(cls, provider_type: AIProvider) -> Dict[str, bool]:
        """Get capabilities for a specific provider
        
        Args:
            provider_type: AI provider enum value
            
        Returns:
            Dictionary with capability flags
        """
        return {
            "supports_embeddings": provider_type in cls._embedding_providers,
            "supports_llm": provider_type in cls._llm_providers,
            "supports_hybrid": provider_type in cls._hybrid_providers,
            "supports_reranking": provider_type in cls._reranking_providers
        }
    
    @classmethod
    def is_provider_available(cls, provider_type: AIProvider) -> bool:
        """Check if a provider type is available
        
        Args:
            provider_type: AI provider enum value
            
        Returns:
            True if provider is registered and available
        """
        return (
            provider_type in cls._embedding_providers or
            provider_type in cls._llm_providers or
            provider_type in cls._hybrid_providers or
            provider_type in cls._reranking_providers
        )
    
    @classmethod
    def list_all_providers(cls) -> Dict[str, List[str]]:
        """List all registered providers by type
        
        Returns:
            Dictionary with provider lists by capability
        """
        return {
            "embedding": [p.value for p in cls._embedding_providers.keys()],
            "llm": [p.value for p in cls._llm_providers.keys()],
            "hybrid": [p.value for p in cls._hybrid_providers.keys()],
            "reranking": [p.value for p in cls._reranking_providers.keys()]
        }
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered providers (mainly for testing)"""
        cls._embedding_providers.clear()
        cls._llm_providers.clear()
        cls._hybrid_providers.clear()
        cls._reranking_providers.clear()
        logger.info("Cleared AI provider registry")


def _register_providers() -> None:
    """Register all available providers"""
    logger.info("Registering AI providers...")
    
    # Try to register OpenAI provider
    try:
        from .providers.openai_provider import OpenAIProvider
        AIProviderFactory.register_hybrid_provider(AIProvider.OPENAI, OpenAIProvider)
    except ImportError as e:
        logger.debug(f"OpenAI provider not available: {e}")
    
    # Try to register Ollama provider
    try:
        from .providers.ollama_provider import OllamaProvider
        AIProviderFactory.register_hybrid_provider(AIProvider.OLLAMA, OllamaProvider)
    except ImportError as e:
        logger.debug(f"Ollama provider not available: {e}")
    
    # Try to register Anthropic provider (LLM only)
    try:
        from .providers.anthropic_provider import AnthropicProvider
        AIProviderFactory.register_llm_provider(AIProvider.ANTHROPIC, AnthropicProvider)
    except ImportError as e:
        logger.debug(f"Anthropic provider not available: {e}")
    
    # Try to register Cohere provider
    try:
        from .providers.cohere_provider import CohereProvider
        AIProviderFactory.register_hybrid_provider(AIProvider.COHERE, CohereProvider)
    except ImportError as e:
        logger.debug(f"Cohere provider not available: {e}")
    
    # Try to register HuggingFace provider
    try:
        from .providers.huggingface_provider import HuggingFaceRerankingProvider
        AIProviderFactory.register_reranking_provider(AIProvider.HUGGINGFACE, HuggingFaceRerankingProvider)
    except ImportError as e:
        logger.debug(f"HuggingFace reranking provider not available: {e}")

    # Try to register vLLM provider (hybrid + vision)
    try:
        from .providers.vllm_provider import VLLMProvider
        AIProviderFactory.register_hybrid_provider(AIProvider.VLLM, VLLMProvider)
        logger.info("vLLM provider registered successfully (hybrid + vision)")
    except ImportError as e:
        logger.debug(f"vLLM provider not available: {e}")

    # Register enhanced providers with reranking capabilities
    try:
        from .providers.openai_provider import OpenAIProvider
        from .providers.ollama_provider import OllamaProvider
        # These are already registered as hybrid providers above, but they now support reranking too
        AIProviderFactory.register_reranking_provider(AIProvider.OPENAI, OpenAIProvider)
        AIProviderFactory.register_reranking_provider(AIProvider.OLLAMA, OllamaProvider)
    except ImportError as e:
        logger.debug(f"Enhanced providers not available for reranking: {e}")

    logger.info(f"AI provider registration complete. Available providers: {AIProviderFactory.list_all_providers()}")


# Register providers on module import
_register_providers()