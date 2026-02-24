"""Configuration management for AI providers"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from .base import AIProvider

logger = logging.getLogger(__name__)


@dataclass
class AIProviderConfig:
    """Configuration for AI providers"""
    provider: AIProvider
    config: Dict[str, Any]
    
    @classmethod
    def from_env(
        cls, 
        provider_type: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        llm_provider: Optional[str] = None
    ) -> 'AIProviderConfig':
        """Create configuration from environment variables
        
        Args:
            provider_type: Single provider type (for backwards compatibility)
            embedding_provider: Specific embedding provider
            llm_provider: Specific LLM provider
            
        Returns:
            AIProviderConfig instance
            
        Raises:
            ValueError: If provider configuration is invalid
        """
        # Handle backwards compatibility
        if provider_type:
            provider_name = provider_type.lower()
        elif embedding_provider and llm_provider:
            # For now, if both are specified, use the embedding provider
            # In future, we might support separate configs for each
            provider_name = embedding_provider.lower()
        elif embedding_provider:
            provider_name = embedding_provider.lower()
        elif llm_provider:
            provider_name = llm_provider.lower()
        else:
            provider_name = os.getenv("AI_PROVIDER", "openai").lower()
        
        try:
            provider = AIProvider(provider_name)
        except ValueError:
            available_providers = [p.value for p in AIProvider]
            raise ValueError(
                f"Unsupported AI provider: {provider_name}. "
                f"Available providers: {available_providers}"
            )
        
        config = cls._get_provider_config(provider)
        
        # Add reranking configuration
        reranking_config = cls._get_reranking_config()
        config.update(reranking_config)
        
        AIProviderConfig._validate_config(provider, config)
        
        return cls(provider=provider, config=config)
    
    @staticmethod
    def _get_provider_config(provider: AIProvider) -> Dict[str, Any]:
        """Get provider-specific configuration from environment
        
        Args:
            provider: AI provider type
            
        Returns:
            Dictionary with provider configuration
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider == AIProvider.OPENAI:
            return {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_BASE_URL"),
                "organization": os.getenv("OPENAI_ORGANIZATION"),
                "default_embedding_model": os.getenv(
                    "OPENAI_EMBEDDING_MODEL", 
                    "text-embedding-3-small"
                ),
                "default_llm_model": os.getenv(
                    "OPENAI_LLM_MODEL", 
                    "gpt-3.5-turbo"
                ),
                "embedding_dimensions": int(os.getenv(
                    "OPENAI_EMBEDDING_DIMENSIONS", 
                    "1536"
                )),
                "max_batch_size": int(os.getenv(
                    "OPENAI_MAX_BATCH_SIZE", 
                    "100"
                )),
                "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("OPENAI_TIMEOUT", "30.0"))
            }
        
        elif provider == AIProvider.OLLAMA:
            return {
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "default_embedding_model": os.getenv(
                    "OLLAMA_EMBEDDING_MODEL", 
                    "nomic-embed-text"
                ),
                "default_llm_model": os.getenv(
                    "OLLAMA_LLM_MODEL", 
                    "llama2"
                ),
                "embedding_dimensions": int(os.getenv(
                    "OLLAMA_EMBEDDING_DIMENSIONS", 
                    "768"
                )),
                "max_batch_size": int(os.getenv(
                    "OLLAMA_MAX_BATCH_SIZE", 
                    "10"
                )),
                "max_retries": int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("OLLAMA_TIMEOUT", "120.0")),
                "pull_models": os.getenv("OLLAMA_PULL_MODELS", "false").lower() == "true"
            }
        
        elif provider == AIProvider.ANTHROPIC:
            return {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "base_url": os.getenv("ANTHROPIC_BASE_URL"),
                "default_llm_model": os.getenv(
                    "ANTHROPIC_LLM_MODEL", 
                    "claude-3-sonnet-20240229"
                ),
                "max_retries": int(os.getenv("ANTHROPIC_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("ANTHROPIC_TIMEOUT", "60.0"))
            }
        
        elif provider == AIProvider.COHERE:
            return {
                "api_key": os.getenv("COHERE_API_KEY"),
                "base_url": os.getenv("COHERE_BASE_URL"),
                "default_embedding_model": os.getenv(
                    "COHERE_EMBEDDING_MODEL", 
                    "embed-english-v3.0"
                ),
                "default_llm_model": os.getenv(
                    "COHERE_LLM_MODEL", 
                    "command"
                ),
                "embedding_dimensions": int(os.getenv(
                    "COHERE_EMBEDDING_DIMENSIONS", 
                    "1024"
                )),
                "max_batch_size": int(os.getenv(
                    "COHERE_MAX_BATCH_SIZE", 
                    "96"
                )),
                "max_retries": int(os.getenv("COHERE_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("COHERE_TIMEOUT", "30.0"))
            }
        
        elif provider == AIProvider.HUGGINGFACE:
            return {
                "api_key": os.getenv("HUGGINGFACE_API_KEY"),
                "base_url": os.getenv(
                    "HUGGINGFACE_BASE_URL",
                    "https://api-inference.huggingface.co"
                ),
                "default_embedding_model": os.getenv(
                    "HUGGINGFACE_EMBEDDING_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                "default_llm_model": os.getenv(
                    "HUGGINGFACE_LLM_MODEL",
                    "microsoft/DialoGPT-medium"
                ),
                "embedding_dimensions": int(os.getenv(
                    "HUGGINGFACE_EMBEDDING_DIMENSIONS",
                    "384"
                )),
                "max_batch_size": int(os.getenv(
                    "HUGGINGFACE_MAX_BATCH_SIZE",
                    "32"
                )),
                "max_retries": int(os.getenv("HUGGINGFACE_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("HUGGINGFACE_TIMEOUT", "60.0"))
            }

        elif provider == AIProvider.VLLM:
            return {
                "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),
                "text_model": os.getenv(
                    "VLLM_TEXT_MODEL",
                    "meta-llama/Llama-3.1-8B-Instruct"
                ),
                "default_embedding_model": os.getenv(
                    "VLLM_EMBEDDING_MODEL",
                    "BAAI/bge-large-en-v1.5"
                ),
                "default_llm_model": os.getenv(
                    "VLLM_TEXT_MODEL",
                    "meta-llama/Llama-3.1-8B-Instruct"
                ),
                "vision_model": os.getenv(
                    "VLLM_VISION_MODEL",
                    "llava-hf/llava-v1.6-mistral-7b-hf"
                ),
                "embedding_dimensions": int(os.getenv(
                    "VLLM_EMBEDDING_DIMENSIONS",
                    "1024"
                )),
                "max_batch_size": int(os.getenv(
                    "VLLM_MAX_BATCH_SIZE",
                    "32"
                )),
                "max_retries": int(os.getenv("VLLM_MAX_RETRIES", "3")),
                "timeout": float(os.getenv("VLLM_TIMEOUT", "120.0")),
                "vision_enabled": os.getenv("VLLM_VISION_ENABLED", "true").lower() == "true"
            }

        else:
            raise ValueError(f"Unknown AI provider: {provider}")
    
    @staticmethod
    def _get_reranking_config() -> Dict[str, Any]:
        """Get reranking configuration from environment variables
        
        Returns:
            Dictionary with reranking configuration
        """
        # General reranking settings
        use_reranking = os.getenv("USE_RERANKING", "false").lower() == "true"
        reranking_provider = os.getenv("RERANKING_PROVIDER", "").lower()
        reranking_model = os.getenv("RERANKING_MODEL", "")
        
        config = {
            "use_reranking": use_reranking,
            "reranking_provider": reranking_provider,
            "reranking_model": reranking_model,
            "reranking_max_results": int(os.getenv("RERANKING_MAX_RESULTS", "100")),
            "reranking_timeout": float(os.getenv("RERANKING_TIMEOUT", "30.0"))
        }
        
        # Provider-specific reranking models
        if reranking_provider == "ollama":
            config.update({
                "default_reranking_model": reranking_model or "bge-reranker-base",
                "reranking_models": [
                    "bge-reranker-base", 
                    "bge-reranker-large",
                    "ms-marco-MiniLM-L-6-v2"
                ]
            })
        elif reranking_provider == "openai":
            config.update({
                "default_reranking_model": "similarity-based",
                "reranking_models": ["similarity-based"]
            })
        elif reranking_provider == "huggingface":
            config.update({
                "default_reranking_model": reranking_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "reranking_models": [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                    "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                ],
                "huggingface_token": os.getenv("HUGGINGFACE_TOKEN")
            })
        
        return config
    
    @staticmethod
    def _validate_config(provider: AIProvider, config: Dict[str, Any]) -> None:
        """Validate provider configuration
        
        Args:
            provider: AI provider type
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Define required keys for each provider
        required_keys = {
            AIProvider.OPENAI: ["api_key"],
            AIProvider.OLLAMA: ["base_url"],
            AIProvider.ANTHROPIC: ["api_key"],
            AIProvider.COHERE: ["api_key"],
            AIProvider.HUGGINGFACE: [],  # API key is optional for public models
            AIProvider.VLLM: ["base_url"]  # API key is optional (defaults to "EMPTY")
        }
        
        # Check required keys
        missing_keys = []
        for key in required_keys.get(provider, []):
            if not config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(
                f"Missing required configuration for {provider.value}: {missing_keys}"
            )
        
        # Validate specific configurations
        AIProviderConfig._validate_provider_specific_config(provider, config)
        
        # Validate reranking configuration
        AIProviderConfig._validate_reranking_config(config)
    
    @staticmethod
    def _validate_provider_specific_config(provider: AIProvider, config: Dict[str, Any]) -> None:
        """Validate provider-specific configuration details
        
        Args:
            provider: AI provider type
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate common configurations
        if "max_batch_size" in config:
            max_batch_size = config["max_batch_size"]
            if not isinstance(max_batch_size, int) or max_batch_size <= 0:
                raise ValueError("max_batch_size must be a positive integer")
        
        if "max_retries" in config:
            max_retries = config["max_retries"]
            if not isinstance(max_retries, int) or max_retries < 0:
                raise ValueError("max_retries must be a non-negative integer")
        
        if "timeout" in config:
            timeout = config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError("timeout must be a positive number")
        
        if "embedding_dimensions" in config:
            dimensions = config["embedding_dimensions"]
            if not isinstance(dimensions, int) or dimensions <= 0:
                raise ValueError("embedding_dimensions must be a positive integer")
        
        # Provider-specific validations
        if provider == AIProvider.OPENAI:
            AIProviderConfig._validate_openai_config(config)
        elif provider == AIProvider.OLLAMA:
            AIProviderConfig._validate_ollama_config(config)
        elif provider == AIProvider.ANTHROPIC:
            AIProviderConfig._validate_anthropic_config(config)
        elif provider == AIProvider.COHERE:
            AIProviderConfig._validate_cohere_config(config)
        elif provider == AIProvider.HUGGINGFACE:
            AIProviderConfig._validate_huggingface_config(config)
        elif provider == AIProvider.VLLM:
            AIProviderConfig._validate_vllm_config(config)
    
    @staticmethod
    def _validate_openai_config(config: Dict[str, Any]) -> None:
        """Validate OpenAI-specific configuration"""
        api_key = config.get("api_key")
        if api_key and not api_key.startswith(("sk-", "sk-proj-")):
            logger.warning("OpenAI API key does not start with expected prefix")
        
        # Validate model names
        embedding_model = config.get("default_embedding_model")
        if embedding_model and not any(embedding_model.startswith(prefix) for prefix in [
            "text-embedding-", "text-similarity-", "text-search-"
        ]):
            logger.warning(f"Unusual OpenAI embedding model: {embedding_model}")
    
    @staticmethod
    def _validate_ollama_config(config: Dict[str, Any]) -> None:
        """Validate Ollama-specific configuration"""
        base_url = config.get("base_url")
        if base_url and not base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must start with http:// or https://")
        
        # Validate reasonable embedding dimensions for Ollama models
        dimensions = config.get("embedding_dimensions", 0)
        if dimensions > 0 and dimensions < 64:
            logger.warning(f"Unusually small embedding dimensions for Ollama: {dimensions}")
    
    @staticmethod
    def _validate_anthropic_config(config: Dict[str, Any]) -> None:
        """Validate Anthropic-specific configuration"""
        api_key = config.get("api_key")
        if api_key and not api_key.startswith("sk-ant-"):
            logger.warning("Anthropic API key does not start with expected prefix")
    
    @staticmethod
    def _validate_cohere_config(config: Dict[str, Any]) -> None:
        """Validate Cohere-specific configuration"""
        # Cohere batch size should not exceed their API limits
        max_batch_size = config.get("max_batch_size", 0)
        if max_batch_size > 96:
            logger.warning(f"Cohere batch size {max_batch_size} exceeds recommended limit of 96")
    
    @staticmethod
    def _validate_huggingface_config(config: Dict[str, Any]) -> None:
        """Validate HuggingFace-specific configuration"""
        # No specific validations for now, but can be extended
        pass

    @staticmethod
    def _validate_vllm_config(config: Dict[str, Any]) -> None:
        """Validate vLLM-specific configuration"""
        base_url = config.get("base_url")
        if base_url and not base_url.startswith(("http://", "https://")):
            raise ValueError("vLLM base_url must start with http:// or https://")

        # Validate vision configuration
        vision_enabled = config.get("vision_enabled", True)
        if vision_enabled:
            vision_model = config.get("vision_model")
            if not vision_model:
                logger.warning("Vision enabled but no vision model specified")

        # Validate embedding dimensions
        dimensions = config.get("embedding_dimensions", 0)
        if dimensions > 0 and dimensions < 64:
            logger.warning(f"Unusually small embedding dimensions for vLLM: {dimensions}")
    
    @staticmethod
    def _validate_reranking_config(config: Dict[str, Any]) -> None:
        """Validate reranking configuration
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If reranking configuration is invalid
        """
        use_reranking = config.get("use_reranking", False)
        if not use_reranking:
            return  # Skip validation if reranking is disabled
        
        reranking_provider = config.get("reranking_provider", "")
        if not reranking_provider:
            raise ValueError("RERANKING_PROVIDER must be specified when USE_RERANKING=true")
        
        # Validate reranking provider
        valid_reranking_providers = ["ollama", "openai", "huggingface"]
        if reranking_provider not in valid_reranking_providers:
            raise ValueError(
                f"Invalid RERANKING_PROVIDER '{reranking_provider}'. "
                f"Valid options: {valid_reranking_providers}"
            )
        
        # Validate max results count
        max_results = config.get("reranking_max_results", 100)
        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("RERANKING_MAX_RESULTS must be a positive integer")
        
        if max_results > 1000:
            logger.warning(f"Large RERANKING_MAX_RESULTS value: {max_results}")
        
        # Validate timeout
        timeout = config.get("reranking_timeout", 30.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("RERANKING_TIMEOUT must be a positive number")
        
        # Provider-specific reranking validation
        if reranking_provider == "huggingface":
            reranking_model = config.get("default_reranking_model", "")
            if reranking_model and not reranking_model.startswith("cross-encoder/"):
                logger.warning(
                    f"HuggingFace reranking model '{reranking_model}' "
                    "doesn't follow expected naming pattern"
                )
        
        elif reranking_provider == "ollama":
            reranking_model = config.get("default_reranking_model", "")
            expected_ollama_models = {"bge-reranker-base", "bge-reranker-large", "ms-marco-MiniLM-L-6-v2"}
            if reranking_model and reranking_model not in expected_ollama_models:
                logger.warning(
                    f"Ollama reranking model '{reranking_model}' not in common models. "
                    f"Expected: {expected_ollama_models}"
                )
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get configuration specific to embedding functionality"""
        embedding_keys = [
            "default_embedding_model", "embedding_dimensions", 
            "max_batch_size", "api_key", "base_url", "timeout", "max_retries"
        ]
        return {k: v for k, v in self.config.items() if k in embedding_keys}
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get configuration specific to LLM functionality"""
        llm_keys = [
            "default_llm_model", "api_key", "base_url", 
            "timeout", "max_retries", "organization"
        ]
        return {k: v for k, v in self.config.items() if k in llm_keys}
    
    def get_reranking_config(self) -> Dict[str, Any]:
        """Get configuration specific to reranking functionality"""
        reranking_keys = [
            "use_reranking", "reranking_provider", "reranking_model",
            "default_reranking_model", "reranking_models", "reranking_max_results",
            "reranking_timeout", "huggingface_token"
        ]
        return {k: v for k, v in self.config.items() if k in reranking_keys and v is not None}
    
    def supports_embeddings(self) -> bool:
        """Check if provider supports embedding generation"""
        embedding_providers = {
            AIProvider.OPENAI, AIProvider.OLLAMA, 
            AIProvider.COHERE, AIProvider.HUGGINGFACE
        }
        return self.provider in embedding_providers
    
    def supports_llm(self) -> bool:
        """Check if provider supports LLM generation"""
        # All current providers support LLM
        return True
    
    def supports_reranking(self) -> bool:
        """Check if provider supports reranking functionality"""
        use_reranking = self.config.get("use_reranking", False)
        reranking_provider = self.config.get("reranking_provider", "")
        
        if not use_reranking or not reranking_provider:
            return False
        
        # Check if the configured reranking provider matches this provider
        # Or if it's a different provider (cross-provider reranking)
        reranking_providers = {"ollama", "openai", "huggingface"}
        return reranking_provider in reranking_providers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for the provider"""
        info = {
            "provider": self.provider.value,
            "supports_embeddings": self.supports_embeddings(),
            "supports_llm": self.supports_llm(),
            "supports_reranking": self.supports_reranking()
        }
        
        if self.supports_embeddings():
            info.update({
                "default_embedding_model": self.config.get("default_embedding_model"),
                "embedding_dimensions": self.config.get("embedding_dimensions")
            })
        
        if self.supports_llm():
            info["default_llm_model"] = self.config.get("default_llm_model")
        
        if self.supports_reranking():
            info.update({
                "reranking_provider": self.config.get("reranking_provider"),
                "default_reranking_model": self.config.get("default_reranking_model"),
                "reranking_models": self.config.get("reranking_models", []),
                "reranking_max_results": self.config.get("reranking_max_results")
            })
        
        return info