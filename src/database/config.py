import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .base import DatabaseProvider

@dataclass
class DatabaseConfig:
    provider: DatabaseProvider
    config: Dict[str, Any]
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        provider_name = os.getenv("VECTOR_DB_PROVIDER", "supabase").lower()
        
        try:
            provider = DatabaseProvider(provider_name)
        except ValueError:
            raise ValueError(f"Unsupported database provider: {provider_name}")
        
        config = cls._get_provider_config(provider)
        cls._validate_config(provider, config)
        
        return cls(provider=provider, config=config)
    
    @staticmethod
    def _get_provider_config(provider: DatabaseProvider) -> Dict[str, Any]:
        if provider == DatabaseProvider.SUPABASE:
            return {
                "url": os.getenv("SUPABASE_URL"),
                "service_key": os.getenv("SUPABASE_SERVICE_KEY")
            }
        elif provider == DatabaseProvider.PINECONE:
            return {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "environment": os.getenv("PINECONE_ENVIRONMENT"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "crawl4ai-rag")
            }
        elif provider == DatabaseProvider.WEAVIATE:
            return {
                "url": os.getenv("WEAVIATE_URL"),
                "api_key": os.getenv("WEAVIATE_API_KEY"),
                "class_name": os.getenv("WEAVIATE_CLASS_NAME", "Document")
            }
        elif provider == DatabaseProvider.SQLITE:
            return {
                "db_path": os.getenv("SQLITE_DB_PATH", "./vector_db.sqlite"),
                "embedding_dimension": int(os.getenv("EMBEDDING_DIMENSION", "1536"))
            }
        elif provider == DatabaseProvider.NEO4J_VECTOR:
            return {
                "uri": os.getenv("NEO4J_URI"),
                "user": os.getenv("NEO4J_USER"),
                "password": os.getenv("NEO4J_PASSWORD"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j")
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