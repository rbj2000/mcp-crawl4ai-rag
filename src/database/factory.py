from typing import Dict, Any
from .base import VectorDatabaseProvider, DatabaseProvider

class DatabaseProviderFactory:
    """Factory for creating database provider instances"""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, provider_type: DatabaseProvider, provider_class):
        """Register a provider class for a given type"""
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: DatabaseProvider, 
        config: Dict[str, Any]
    ) -> VectorDatabaseProvider:
        if provider_type not in cls._providers:
            raise ValueError(f"Unsupported database provider: {provider_type}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list[DatabaseProvider]:
        """Get list of available providers"""
        return list(cls._providers.keys())

# Import and register providers when they become available
def _register_providers():
    """Register all available providers"""
    try:
        from .providers.supabase_provider import SupabaseProvider
        DatabaseProviderFactory.register_provider(DatabaseProvider.SUPABASE, SupabaseProvider)
    except ImportError:
        pass
    
    try:
        from .providers.sqlite_provider import SQLiteProvider
        DatabaseProviderFactory.register_provider(DatabaseProvider.SQLITE, SQLiteProvider)
    except ImportError:
        pass
    
    try:
        from .providers.pinecone_provider import PineconeProvider
        DatabaseProviderFactory.register_provider(DatabaseProvider.PINECONE, PineconeProvider)
    except ImportError:
        pass
    
    try:
        from .providers.weaviate_provider import WeaviateProvider
        DatabaseProviderFactory.register_provider(DatabaseProvider.WEAVIATE, WeaviateProvider)
    except ImportError:
        pass
    
    try:
        from .providers.neo4j_provider import Neo4jVectorProvider
        DatabaseProviderFactory.register_provider(DatabaseProvider.NEO4J_VECTOR, Neo4jVectorProvider)
    except ImportError:
        pass

# Register providers on module import
_register_providers()