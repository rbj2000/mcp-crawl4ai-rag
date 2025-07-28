import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .base import DatabaseProvider

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    provider: DatabaseProvider
    config: Dict[str, Any]
    embedding_dimension: int = 1536  # Default OpenAI dimension
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_embedding_dimension()
    
    def _validate_embedding_dimension(self) -> None:
        """Validate embedding dimension"""
        if not isinstance(self.embedding_dimension, int):
            raise ValueError("Embedding dimension must be an integer")
        
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        # Common embedding dimensions validation
        common_dimensions = [
            384,   # sentence-transformers/all-MiniLM-L6-v2
            512,   # sentence-transformers/all-mpnet-base-v2
            768,   # BERT base models
            1024,  # sentence-transformers/all-mpnet-base-v2
            1536,  # OpenAI text-embedding-3-small
            3072,  # OpenAI text-embedding-3-large
        ]
        
        if self.embedding_dimension not in common_dimensions:
            logger.warning(
                f"Embedding dimension {self.embedding_dimension} is not a common "
                f"dimension. Common dimensions are: {common_dimensions}"
            )
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        provider_name = os.getenv("VECTOR_DB_PROVIDER", "supabase").lower()
        
        try:
            provider = DatabaseProvider(provider_name)
        except ValueError:
            available_providers = [p.value for p in DatabaseProvider]
            raise ValueError(f"Unsupported database provider: {provider_name}. Available: {available_providers}")
        
        # Get embedding dimension from environment or AI provider
        embedding_dimension = cls._get_embedding_dimension()
        
        config = cls._get_provider_config(provider, embedding_dimension)
        cls._validate_config(provider, config)
        
        return cls(provider=provider, config=config, embedding_dimension=embedding_dimension)
    
    @classmethod 
    def _get_embedding_dimension(cls) -> int:
        """Get embedding dimension from environment or auto-detect from AI provider"""
        # First check if explicitly set
        if os.getenv("EMBEDDING_DIMENSION"):
            try:
                return int(os.getenv("EMBEDDING_DIMENSION"))
            except ValueError:
                raise ValueError("EMBEDDING_DIMENSION must be a valid integer")
        
        # Try to auto-detect from AI provider configuration
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower()
        
        if ai_provider == "openai":
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            return cls._get_openai_model_dimensions(embedding_model)
        elif ai_provider == "ollama":
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
            return cls._get_ollama_model_dimensions(embedding_model)
        else:
            logger.warning(f"Unknown AI provider {ai_provider}, using default dimension 1536")
            return 1536
    
    @classmethod
    def _get_openai_model_dimensions(cls, model: str) -> int:
        """Get dimensions for OpenAI embedding models"""
        openai_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return openai_dimensions.get(model, 1536)
    
    @classmethod
    def _get_ollama_model_dimensions(cls, model: str) -> int:
        """Get dimensions for Ollama embedding models (requires model introspection)"""
        # Common Ollama embedding model dimensions
        ollama_dimensions = {
            "nomic-embed-text:latest": 768,
            "all-minilm:latest": 384,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "bge-m3": 1024,  # BGE-M3 model
        }
        return ollama_dimensions.get(model, 768)  # Default for most Ollama models
    
    @staticmethod
    def _get_provider_config(provider: DatabaseProvider, embedding_dimension: int) -> Dict[str, Any]:
        if provider == DatabaseProvider.SUPABASE:
            return {
                "url": os.getenv("SUPABASE_URL"),
                "service_key": os.getenv("SUPABASE_SERVICE_KEY"),
                "embedding_dimension": embedding_dimension
            }
        elif provider == DatabaseProvider.PINECONE:
            return {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "environment": os.getenv("PINECONE_ENVIRONMENT"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "crawl4ai-rag"),
                "embedding_dimension": embedding_dimension
            }
        elif provider == DatabaseProvider.WEAVIATE:
            return {
                "url": os.getenv("WEAVIATE_URL"),
                "api_key": os.getenv("WEAVIATE_API_KEY"),
                "class_name": os.getenv("WEAVIATE_CLASS_NAME", "Document"),
                "embedding_dimension": embedding_dimension
            }
        elif provider == DatabaseProvider.SQLITE:
            return {
                "db_path": os.getenv("SQLITE_DB_PATH", "./vector_db.sqlite"),
                "embedding_dimension": embedding_dimension
            }
        elif provider == DatabaseProvider.NEO4J_VECTOR:
            return {
                "uri": os.getenv("NEO4J_URI"),
                "user": os.getenv("NEO4J_USER"),
                "password": os.getenv("NEO4J_PASSWORD"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
                "embedding_dimension": embedding_dimension
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def _validate_config(provider: DatabaseProvider, config: Dict[str, Any]) -> None:
        required_keys = {
            DatabaseProvider.SUPABASE: ["url", "service_key"],
            DatabaseProvider.PINECONE: ["api_key", "environment", "index_name"],
            DatabaseProvider.WEAVIATE: ["url"],
            DatabaseProvider.SQLITE: ["db_path"],
            DatabaseProvider.NEO4J_VECTOR: ["uri", "user", "password"]
        }
        
        missing_keys = []
        for key in required_keys[provider]:
            if not config.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration for {provider.value}: {missing_keys}")