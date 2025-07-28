# Database-Agnostic Implementation Plan

## Overview

This document provides a detailed implementation plan for transforming the mcp-crawl4ai-rag system into a database-agnostic platform that supports multiple vector database backends.

## Phase 1: Database Abstraction Layer (Week 1-2)

### 1.1 Create Abstract Interfaces

**File**: `src/database/base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class DatabaseProvider(Enum):
    SUPABASE = "supabase"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"
    NEO4J_VECTOR = "neo4j_vector"

@dataclass
class VectorSearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    url: str
    chunk_number: int
    source_id: str

@dataclass
class DocumentChunk:
    url: str
    chunk_number: int
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    source_id: str

@dataclass
class CodeExample:
    url: str
    chunk_number: int
    content: str
    summary: str
    metadata: Dict[str, Any]
    embedding: List[float]
    source_id: str

@dataclass
class SourceInfo:
    source_id: str
    summary: str
    total_words: int
    created_at: str
    updated_at: str

class VectorDatabaseProvider(ABC):
    """Abstract base class for vector database providers"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection and schema"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the database connection"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive"""
        pass
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        batch_size: int = 20
    ) -> None:
        """Add document chunks to the database"""
        pass
    
    @abstractmethod
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def add_code_examples(
        self, 
        examples: List[CodeExample], 
        batch_size: int = 20
    ) -> None:
        """Add code examples to the database"""
        pass
    
    @abstractmethod
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar code examples"""
        pass
    
    @abstractmethod
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        pass
    
    @abstractmethod
    async def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int
    ) -> None:
        """Update source information"""
        pass
    
    @abstractmethod
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        pass
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        Default implementation falls back to vector search only.
        Providers can override this for true hybrid search.
        """
        return await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    async def hybrid_search_code_examples(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search for code examples.
        Default implementation falls back to vector search only.
        """
        return await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
```

### 1.2 Create Configuration Management

**File**: `src/database/config.py`

```python
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
```

### 1.3 Create Provider Factory

**File**: `src/database/factory.py`

```python
from typing import Dict, Any
from .base import VectorDatabaseProvider, DatabaseProvider
from .providers.supabase_provider import SupabaseProvider
from .providers.pinecone_provider import PineconeProvider
from .providers.weaviate_provider import WeaviateProvider
from .providers.sqlite_provider import SQLiteProvider
from .providers.neo4j_provider import Neo4jVectorProvider

class DatabaseProviderFactory:
    """Factory for creating database provider instances"""
    
    _providers = {
        DatabaseProvider.SUPABASE: SupabaseProvider,
        DatabaseProvider.PINECONE: PineconeProvider,
        DatabaseProvider.WEAVIATE: WeaviateProvider,
        DatabaseProvider.SQLITE: SQLiteProvider,
        DatabaseProvider.NEO4J_VECTOR: Neo4jVectorProvider,
    }
    
    @classmethod
    def create_provider(
        self, 
        provider_type: DatabaseProvider, 
        config: Dict[str, Any]
    ) -> VectorDatabaseProvider:
        if provider_type not in self._providers:
            raise ValueError(f"Unsupported database provider: {provider_type}")
        
        provider_class = self._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list[DatabaseProvider]:
        """Get list of available providers"""
        return list(cls._providers.keys())
```

### 1.4 Migrate Supabase Provider

**File**: `src/database/providers/supabase_provider.py`

```python
import json
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from supabase import create_client, Client
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

class SupabaseProvider(VectorDatabaseProvider):
    """Supabase/pgvector implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[Client] = None
    
    async def initialize(self) -> None:
        """Initialize the Supabase client"""
        self.client = create_client(
            self.config["url"],
            self.config["service_key"]
        )
        
        # Test connection
        await self.health_check()
    
    async def close(self) -> None:
        """Close the database connection"""
        # Supabase client doesn't need explicit closing
        self.client = None
    
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive"""
        try:
            # Simple query to test connection
            result = self.client.table("crawled_pages").select("count", count="exact").limit(0).execute()
            return True
        except Exception as e:
            print(f"Supabase health check failed: {e}")
            return False
    
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        batch_size: int = 20
    ) -> None:
        """Add document chunks to Supabase"""
        if not documents:
            return
        
        # Get unique URLs to delete existing records
        unique_urls = list(set(doc.url for doc in documents))
        
        # Delete existing records
        try:
            if unique_urls:
                self.client.table("crawled_pages").delete().in_("url", unique_urls).execute()
        except Exception as e:
            print(f"Batch delete failed: {e}")
            # Fallback: delete one by one
            for url in unique_urls:
                try:
                    self.client.table("crawled_pages").delete().eq("url", url).execute()
                except Exception as inner_e:
                    print(f"Error deleting record for URL {url}: {inner_e}")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_documents = documents[i:batch_end]
            
            batch_data = []
            for doc in batch_documents:
                data = {
                    "url": doc.url,
                    "chunk_number": doc.chunk_number,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "source_id": doc.source_id,
                    "embedding": doc.embedding
                }
                batch_data.append(data)
            
            # Insert with retry logic
            await self._insert_with_retry(
                table="crawled_pages", 
                data=batch_data, 
                max_retries=3
            )
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = self.client.rpc('match_crawled_pages', params).execute()
            
            return [
                VectorSearchResult(
                    id=str(row.get('id', '')),
                    content=row.get('content', ''),
                    metadata=row.get('metadata', {}),
                    similarity=row.get('similarity', 0.0),
                    url=row.get('url', ''),
                    chunk_number=row.get('chunk_number', 0),
                    source_id=row.get('source_id', '')
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def add_code_examples(
        self, 
        examples: List[CodeExample], 
        batch_size: int = 20
    ) -> None:
        """Add code examples to Supabase"""
        if not examples:
            return
        
        # Delete existing records for these URLs
        unique_urls = list(set(ex.url for ex in examples))
        for url in unique_urls:
            try:
                self.client.table('code_examples').delete().eq('url', url).execute()
            except Exception as e:
                print(f"Error deleting existing code examples for {url}: {e}")
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch_end = min(i + batch_size, len(examples))
            batch_examples = examples[i:batch_end]
            
            batch_data = []
            for ex in batch_examples:
                data = {
                    'url': ex.url,
                    'chunk_number': ex.chunk_number,
                    'content': ex.content,
                    'summary': ex.summary,
                    'metadata': ex.metadata,
                    'source_id': ex.source_id,
                    'embedding': ex.embedding
                }
                batch_data.append(data)
            
            await self._insert_with_retry(
                table="code_examples", 
                data=batch_data, 
                max_retries=3
            )
    
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar code examples"""
        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = self.client.rpc('match_code_examples', params).execute()
            
            return [
                VectorSearchResult(
                    id=str(row.get('id', '')),
                    content=row.get('content', ''),
                    metadata=row.get('metadata', {}),
                    similarity=row.get('similarity', 0.0),
                    url=row.get('url', ''),
                    chunk_number=row.get('chunk_number', 0),
                    source_id=row.get('source_id', '')
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error searching code examples: {e}")
            return []
    
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        try:
            result = self.client.from_('sources').select('*').order('source_id').execute()
            
            return [
                SourceInfo(
                    source_id=row.get("source_id"),
                    summary=row.get("summary"),
                    total_words=row.get("total_words", 0),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    async def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int
    ) -> None:
        """Update source information"""
        try:
            # Try to update existing source
            result = self.client.table('sources').update({
                'summary': summary,
                'total_words': word_count,
                'updated_at': 'now()'
            }).eq('source_id', source_id).execute()
            
            # If no rows were updated, insert new source
            if not result.data:
                self.client.table('sources').insert({
                    'source_id': source_id,
                    'summary': summary,
                    'total_words': word_count
                }).execute()
                
        except Exception as e:
            print(f"Error updating source {source_id}: {e}")
    
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        try:
            # Delete from both tables
            self.client.table("crawled_pages").delete().in_("url", urls).execute()
            self.client.table("code_examples").delete().in_("url", urls).execute()
        except Exception as e:
            print(f"Error deleting by URLs: {e}")
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        # Get vector search results
        vector_results = await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        keyword_query_builder = self.client.from_('crawled_pages')\
            .select('id, url, chunk_number, content, metadata, source_id')\
            .ilike('content', f'%{keyword_query}%')
        
        # Apply source filter if provided
        if filter_metadata and 'source' in filter_metadata:
            keyword_query_builder = keyword_query_builder.eq('source_id', filter_metadata['source'])
        
        keyword_response = keyword_query_builder.limit(match_count * 2).execute()
        keyword_results = keyword_response.data if keyword_response.data else []
        
        # Combine results
        return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    async def hybrid_search_code_examples(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform hybrid search for code examples"""
        # Get vector search results
        vector_results = await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        keyword_query_builder = self.client.from_('code_examples')\
            .select('id, url, chunk_number, content, summary, metadata, source_id')\
            .or_(f'content.ilike.%{keyword_query}%,summary.ilike.%{keyword_query}%')
        
        # Apply source filter if provided
        if filter_metadata and 'source' in filter_metadata:
            keyword_query_builder = keyword_query_builder.eq('source_id', filter_metadata['source'])
        
        keyword_response = keyword_query_builder.limit(match_count * 2).execute()
        keyword_results = keyword_response.data if keyword_response.data else []
        
        # Combine results
        return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[VectorSearchResult], 
        keyword_results: List[Dict], 
        match_count: int
    ) -> List[VectorSearchResult]:
        """Combine vector and keyword search results"""
        seen_ids = set()
        combined_results = []
        
        # Convert keyword results to VectorSearchResult format
        keyword_search_results = []
        for kr in keyword_results:
            keyword_search_results.append(VectorSearchResult(
                id=str(kr['id']),
                url=kr['url'],
                chunk_number=kr['chunk_number'],
                content=kr.get('content', ''),
                metadata=kr.get('metadata', {}),
                source_id=kr.get('source_id', ''),
                similarity=0.5  # Default similarity for keyword-only matches
            ))
        
        # Add items that appear in both searches first (boost their scores)
        vector_ids = {r.id for r in vector_results}
        for kr in keyword_search_results:
            if kr.id in vector_ids and kr.id not in seen_ids:
                # Find the vector result to get similarity score
                for vr in vector_results:
                    if vr.id == kr.id:
                        # Boost similarity score for items in both results
                        vr.similarity = min(1.0, vr.similarity * 1.2)
                        combined_results.append(vr)
                        seen_ids.add(kr.id)
                        break
        
        # Add remaining vector results
        for vr in vector_results:
            if vr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(vr)
                seen_ids.add(vr.id)
        
        # Add pure keyword matches if needed
        for kr in keyword_search_results:
            if kr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(kr)
                seen_ids.add(kr.id)
        
        return combined_results[:match_count]
    
    async def _insert_with_retry(
        self, 
        table: str, 
        data: List[Dict[str, Any]], 
        max_retries: int = 3
    ) -> None:
        """Insert data with retry logic"""
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                self.client.table(table).insert(data).execute()
                return
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Try inserting records individually
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in data:
                        try:
                            self.client.table(table).insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(data)} records individually")
```

## Phase 2: Refactor Main Application (Week 2-3)

### 2.1 Update Main MCP Server

**File**: `src/crawl4ai_mcp.py` (major refactoring)

Key changes:
1. Replace direct Supabase imports with database abstraction
2. Update context management
3. Modify all tool functions to use the abstraction layer

```python
# Replace imports
from database.config import DatabaseConfig
from database.factory import DatabaseProviderFactory
from database.base import VectorDatabaseProvider, DocumentChunk, CodeExample

# Update context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    vector_db: VectorDatabaseProvider
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None
    repo_extractor: Optional[Any] = None

# Update lifespan function
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    # ... crawler initialization ...
    
    # Initialize database provider
    db_config = DatabaseConfig.from_env()
    vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
    await vector_db.initialize()
    
    # ... rest of initialization ...
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            vector_db=vector_db,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor
        )
    finally:
        await crawler.__aexit__(None, None, None)
        await vector_db.close()
        # ... cleanup ...
```

### 2.2 Update Utility Functions

**File**: `src/utils_abstracted.py`

```python
# New abstracted utility functions
async def add_documents_to_database(
    vector_db: VectorDatabaseProvider,
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """Add documents to the vector database using abstraction layer"""
    
    # Apply contextual embedding if enabled
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    
    if use_contextual_embeddings:
        processed_contents = []
        for i, content in enumerate(contents):
            url = urls[i]
            full_document = url_to_full_document.get(url, "")
            contextual_text, success = generate_contextual_embedding(full_document, content)
            processed_contents.append(contextual_text)
            if success:
                metadatas[i]["contextual_embedding"] = True
    else:
        processed_contents = contents
    
    # Create embeddings
    embeddings = create_embeddings_batch(processed_contents)
    
    # Convert to DocumentChunk objects
    documents = []
    for i, content in enumerate(processed_contents):
        parsed_url = urlparse(urls[i])
        source_id = parsed_url.netloc or parsed_url.path
        
        doc = DocumentChunk(
            url=urls[i],
            chunk_number=chunk_numbers[i],
            content=content,
            metadata=metadatas[i],
            embedding=embeddings[i],
            source_id=source_id
        )
        documents.append(doc)
    
    # Add to database
    await vector_db.add_documents(documents, batch_size)

async def search_documents_abstracted(
    vector_db: VectorDatabaseProvider,
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = False
) -> List[Dict[str, Any]]:
    """Search documents using the abstraction layer"""
    
    query_embedding = create_embedding(query)
    
    if use_hybrid:
        results = await vector_db.hybrid_search_documents(
            query_embedding=query_embedding,
            keyword_query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    else:
        results = await vector_db.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    # Convert to dictionary format for JSON serialization
    return [
        {
            "id": result.id,
            "url": result.url,
            "content": result.content,
            "metadata": result.metadata,
            "similarity": result.similarity,
            "source_id": result.source_id
        }
        for result in results
    ]
```

## Phase 3: Additional Database Providers (Week 3-4)

### 3.1 Pinecone Provider

**File**: `src/database/providers/pinecone_provider.py`

```python
import asyncio
from typing import List, Dict, Any, Optional
import pinecone
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

class PineconeProvider(VectorDatabaseProvider):
    """Pinecone vector database implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index = None
        self.documents_index_name = config["index_name"]
        self.code_examples_index_name = f"{config['index_name']}-code"
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection"""
        pinecone.init(
            api_key=self.config["api_key"],
            environment=self.config["environment"]
        )
        
        # Create indexes if they don't exist
        await self._ensure_indexes_exist()
        
        self.index = pinecone.Index(self.documents_index_name)
        self.code_index = pinecone.Index(self.code_examples_index_name)
    
    async def close(self) -> None:
        """Close connections"""
        self.index = None
        self.code_index = None
    
    async def health_check(self) -> bool:
        """Check if Pinecone is responsive"""
        try:
            stats = self.index.describe_index_stats()
            return True
        except Exception as e:
            print(f"Pinecone health check failed: {e}")
            return False
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 100) -> None:
        """Add documents to Pinecone"""
        if not documents:
            return
        
        # Delete existing documents by URL first
        unique_urls = list(set(doc.url for doc in documents))
        await self._delete_by_metadata_filter("url", unique_urls)
        
        # Prepare vectors for insertion
        vectors = []
        for doc in documents:
            vector_id = f"{doc.url}#{doc.chunk_number}"
            vectors.append({
                "id": vector_id,
                "values": doc.embedding,
                "metadata": {
                    "url": doc.url,
                    "chunk_number": doc.chunk_number,
                    "content": doc.content[:1000],  # Pinecone metadata size limit
                    "source_id": doc.source_id,
                    "full_content": doc.content,  # Store full content separately
                    **{k: str(v) for k, v in doc.metadata.items()}  # Convert all to strings
                }
            })
        
        # Insert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            await asyncio.sleep(0.1)  # Rate limiting
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search documents in Pinecone"""
        try:
            filter_dict = {}
            if filter_metadata and "source" in filter_metadata:
                filter_dict["source_id"] = {"$eq": filter_metadata["source"]}
            
            results = self.index.query(
                vector=query_embedding,
                top_k=match_count,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            search_results = []
            for match in results.matches:
                metadata = match.metadata
                search_results.append(VectorSearchResult(
                    id=match.id,
                    content=metadata.get("full_content", metadata.get("content", "")),
                    metadata={k: v for k, v in metadata.items() if k not in ["full_content", "content", "url", "source_id", "chunk_number"]},
                    similarity=match.score,
                    url=metadata.get("url", ""),
                    chunk_number=int(metadata.get("chunk_number", 0)),
                    source_id=metadata.get("source_id", "")
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching documents in Pinecone: {e}")
            return []
    
    async def _ensure_indexes_exist(self) -> None:
        """Create indexes if they don't exist"""
        existing_indexes = pinecone.list_indexes()
        
        if self.documents_index_name not in existing_indexes:
            pinecone.create_index(
                name=self.documents_index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        if self.code_examples_index_name not in existing_indexes:
            pinecone.create_index(
                name=self.code_examples_index_name,
                dimension=1536,
                metric="cosine"
            )
    
    async def _delete_by_metadata_filter(self, key: str, values: List[str]) -> None:
        """Delete vectors by metadata filter"""
        for value in values:
            filter_dict = {key: {"$eq": value}}
            self.index.delete(filter=filter_dict)
            if hasattr(self, 'code_index'):
                self.code_index.delete(filter=filter_dict)
    
    # Implement remaining abstract methods...
```

### 3.2 SQLite Provider

**File**: `src/database/providers/sqlite_provider.py`

```python
import sqlite3
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

class SQLiteProvider(VectorDatabaseProvider):
    """Local SQLite with vector similarity implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config["db_path"]
        self.embedding_dimension = config.get("embedding_dimension", 1536)
        self.connection = None
    
    async def initialize(self) -> None:
        """Initialize SQLite database"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        await self._create_tables()
        await self._create_indexes()
    
    async def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def health_check(self) -> bool:
        """Check if database is responsive"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"SQLite health check failed: {e}")
            return False
    
    async def _create_tables(self) -> None:
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Code examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                summary TEXT,
                total_words INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    async def _create_indexes(self) -> None:
        """Create database indexes"""
        cursor = self.connection.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)",
            "CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_examples_url ON code_examples(url)",
            "CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples(source_id)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        """Add documents to SQLite"""
        if not documents:
            return
        
        cursor = self.connection.cursor()
        
        # Delete existing documents by URL
        unique_urls = list(set(doc.url for doc in documents))
        placeholders = ",".join("?" * len(unique_urls))
        cursor.execute(f"DELETE FROM documents WHERE url IN ({placeholders})", unique_urls)
        
        # Prepare data for insertion
        data = []
        for doc in documents:
            embedding_blob = np.array(doc.embedding, dtype=np.float32).tobytes()
            data.append((
                doc.url,
                doc.chunk_number,
                doc.content,
                json.dumps(doc.metadata),
                doc.source_id,
                embedding_blob
            ))
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany("""
                INSERT OR REPLACE INTO documents 
                (url, chunk_number, content, metadata, source_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
        
        self.connection.commit()
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search documents using cosine similarity"""
        cursor = self.connection.cursor()
        
        # Build query with optional filtering
        sql = """
            SELECT id, url, chunk_number, content, metadata, source_id, embedding
            FROM documents
        """
        params = []
        
        if filter_metadata and "source" in filter_metadata:
            sql += " WHERE source_id = ?"
            params.append(filter_metadata["source"])
        
        cursor.execute(sql, params)
        
        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        for row in cursor.fetchall():
            # Deserialize embedding
            doc_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_vec], [doc_embedding])[0][0]
            
            results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=float(similarity),
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
            ))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:match_count]
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Hybrid search combining vector similarity and keyword search"""
        # Get vector search results
        vector_results = await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        cursor = self.connection.cursor()
        sql = """
            SELECT id, url, chunk_number, content, metadata, source_id
            FROM documents
            WHERE content LIKE ?
        """
        params = [f"%{keyword_query}%"]
        
        if filter_metadata and "source" in filter_metadata:
            sql += " AND source_id = ?"
            params.append(filter_metadata["source"])
        
        sql += " LIMIT ?"
        params.append(match_count * 2)
        
        cursor.execute(sql, params)
        
        keyword_results = []
        for row in cursor.fetchall():
            keyword_results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=0.5,  # Default similarity for keyword matches
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
            ))
        
        # Combine results
        return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[VectorSearchResult], 
        keyword_results: List[VectorSearchResult], 
        match_count: int
    ) -> List[VectorSearchResult]:
        """Combine vector and keyword search results"""
        seen_ids = set()
        combined_results = []
        
        # Boost items that appear in both results
        vector_ids = {r.id for r in vector_results}
        for kr in keyword_results:
            if kr.id in vector_ids and kr.id not in seen_ids:
                # Find the vector result and boost its score
                for vr in vector_results:
                    if vr.id == kr.id:
                        vr.similarity = min(1.0, vr.similarity * 1.2)
                        combined_results.append(vr)
                        seen_ids.add(kr.id)
                        break
        
        # Add remaining vector results
        for vr in vector_results:
            if vr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(vr)
                seen_ids.add(vr.id)
        
        # Add remaining keyword results
        for kr in keyword_results:
            if kr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(kr)
                seen_ids.add(kr.id)
        
        return combined_results[:match_count]
    
    # Implement remaining abstract methods...
```

## Phase 4: Update Dependencies and Configuration (Week 4)

### 4.1 Update pyproject.toml

```toml
[project]
name = "crawl4ai-mcp"
version = "0.2.0"
description = "Database-agnostic MCP server for integrating web crawling and RAG into AI agents and AI coding assistants"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "crawl4ai==0.6.2",
    "mcp==1.7.1",
    "openai==1.71.0",
    "dotenv==0.9.9",
    "sentence-transformers>=4.1.0",
    "neo4j>=5.28.1",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
supabase = ["supabase==2.15.1"]
pinecone = ["pinecone-client>=3.0.0"]
weaviate = ["weaviate-client>=4.0.0"]
all = [
    "supabase==2.15.1",
    "pinecone-client>=3.0.0", 
    "weaviate-client>=4.0.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

### 4.2 Update Environment Configuration

**File**: `.env.example`

```bash
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# LLM for summaries and contextual embeddings
MODEL_CHOICE=gpt-4o-mini

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Database Provider Selection
VECTOR_DB_PROVIDER=supabase  # supabase, pinecone, weaviate, sqlite, neo4j_vector

# Supabase Configuration (when VECTOR_DB_PROVIDER=supabase)
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Pinecone Configuration (when VECTOR_DB_PROVIDER=pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=crawl4ai-rag

# Weaviate Configuration (when VECTOR_DB_PROVIDER=weaviate)
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_CLASS_NAME=Document

# SQLite Configuration (when VECTOR_DB_PROVIDER=sqlite)
SQLITE_DB_PATH=./vector_db.sqlite
EMBEDDING_DIMENSION=1536

# Neo4j Vector Configuration (when VECTOR_DB_PROVIDER=neo4j_vector)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

# Knowledge Graph Neo4j Configuration (separate from vector DB)
# Required for hallucination detection features
KG_NEO4J_URI=bolt://localhost:7687
KG_NEO4J_USER=neo4j
KG_NEO4J_PASSWORD=your_neo4j_password
```

### 4.3 Update Docker Configuration

**File**: `Dockerfile`

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies based on provider
ARG VECTOR_DB_PROVIDER=supabase
RUN if [ "$VECTOR_DB_PROVIDER" = "all" ]; then \
        uv pip install --system -e ".[all]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "supabase" ]; then \
        uv pip install --system -e ".[supabase]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "pinecone" ]; then \
        uv pip install --system -e ".[pinecone]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "weaviate" ]; then \
        uv pip install --system -e ".[weaviate]"; \
    else \
        uv pip install --system -e .; \
    fi

# Setup crawl4ai
RUN crawl4ai-setup

# Copy source code
COPY src/ ./src/
COPY knowledge_graphs/ ./knowledge_graphs/

# Set default environment variables
ENV HOST=0.0.0.0
ARG PORT=8051
ENV PORT=$PORT
ENV TRANSPORT=sse
ENV VECTOR_DB_PROVIDER=supabase

EXPOSE $PORT

CMD ["python", "src/crawl4ai_mcp.py"]
```

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  # Supabase deployment
  mcp-crawl4ai-supabase:
    build: 
      context: .
      args:
        VECTOR_DB_PROVIDER: supabase
    environment:
      - VECTOR_DB_PROVIDER=supabase
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_CHOICE=${MODEL_CHOICE}
    ports:
      - "8051:8051"
    profiles: ["supabase"]

  # SQLite deployment (local development)
  mcp-crawl4ai-sqlite:
    build:
      context: .
      args:
        VECTOR_DB_PROVIDER: sqlite
    environment:
      - VECTOR_DB_PROVIDER=sqlite
      - SQLITE_DB_PATH=/data/vector_db.sqlite
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_CHOICE=${MODEL_CHOICE}
    volumes:
      - ./data:/data
    ports:
      - "8052:8051"
    profiles: ["sqlite"]

  # Weaviate deployment
  mcp-crawl4ai-weaviate:
    build:
      context: .
      args:
        VECTOR_DB_PROVIDER: weaviate
    environment:
      - VECTOR_DB_PROVIDER=weaviate
      - WEAVIATE_URL=http://weaviate:8080
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_CHOICE=${MODEL_CHOICE}
    depends_on:
      - weaviate
    ports:
      - "8053:8051"
    profiles: ["weaviate"]

  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - ENABLE_MODULES=text2vec-openai
      - OPENAI_APIKEY=${OPENAI_API_KEY}
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate
    profiles: ["weaviate"]

  # Pinecone deployment
  mcp-crawl4ai-pinecone:
    build:
      context: .
      args:
        VECTOR_DB_PROVIDER: pinecone
    environment:
      - VECTOR_DB_PROVIDER=pinecone
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}
      - PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MODEL_CHOICE=${MODEL_CHOICE}
    ports:
      - "8054:8051"
    profiles: ["pinecone"]

volumes:
  weaviate_data:
```

## Phase 5: Testing and Validation (Week 5)

### 5.1 Create Test Suite

**File**: `tests/test_database_providers.py`

```python
import pytest
import asyncio
import os
from typing import List
from src.database.base import VectorDatabaseProvider, DocumentChunk, CodeExample
from src.database.config import DatabaseConfig
from src.database.factory import DatabaseProviderFactory

class DatabaseProviderTestSuite:
    """Comprehensive test suite for all database providers"""
    
    @pytest.fixture
    async def provider(self, provider_config) -> VectorDatabaseProvider:
        """Create and initialize a database provider"""
        provider = DatabaseProviderFactory.create_provider(
            provider_config["type"], 
            provider_config["config"]
        )
        await provider.initialize()
        yield provider
        await provider.close()
    
    async def test_health_check(self, provider):
        """Test database health check"""
        assert await provider.health_check() == True
    
    async def test_document_crud_operations(self, provider):
        """Test basic document CRUD operations"""
        # Create test documents
        test_docs = [
            DocumentChunk(
                url="https://test.com/doc1",
                chunk_number=0,
                content="This is a test document for vector search",
                metadata={"test": True, "category": "docs"},
                embedding=[0.1] * 1536,  # Mock embedding
                source_id="test.com"
            ),
            DocumentChunk(
                url="https://test.com/doc2", 
                chunk_number=0,
                content="Another test document with different content",
                metadata={"test": True, "category": "guides"},
                embedding=[0.2] * 1536,
                source_id="test.com"
            )
        ]
        
        # Add documents
        await provider.add_documents(test_docs)
        
        # Search for documents
        results = await provider.search_documents(
            query_embedding=[0.15] * 1536,
            match_count=2
        )
        
        assert len(results) >= 1
        assert any("test document" in result.content for result in results)
        
        # Test filtering
        filtered_results = await provider.search_documents(
            query_embedding=[0.15] * 1536,
            match_count=2,
            filter_metadata={"source": "test.com"}
        )
        
        assert len(filtered_results) >= 1
        assert all(result.source_id == "test.com" for result in filtered_results)
    
    async def test_code_example_operations(self, provider):
        """Test code example operations"""
        test_examples = [
            CodeExample(
                url="https://test.com/code1",
                chunk_number=0,
                content="def hello_world():\n    print('Hello, World!')",
                summary="A simple hello world function",
                metadata={"language": "python", "test": True},
                embedding=[0.3] * 1536,
                source_id="test.com"
            )
        ]
        
        await provider.add_code_examples(test_examples)
        
        results = await provider.search_code_examples(
            query_embedding=[0.35] * 1536,
            match_count=1
        )
        
        assert len(results) >= 1
        assert "hello_world" in results[0].content
    
    async def test_source_management(self, provider):
        """Test source information management"""
        await provider.update_source_info(
            source_id="test.com",
            summary="Test website for unit tests",
            word_count=100
        )
        
        sources = await provider.get_sources()
        test_source = next((s for s in sources if s.source_id == "test.com"), None)
        
        assert test_source is not None
        assert test_source.summary == "Test website for unit tests"
        assert test_source.total_words == 100
    
    async def test_hybrid_search(self, provider):
        """Test hybrid search if supported"""
        if hasattr(provider, 'hybrid_search_documents'):
            # Add test documents first
            test_docs = [
                DocumentChunk(
                    url="https://test.com/hybrid1",
                    chunk_number=0,
                    content="Python is a programming language with great vector support",
                    metadata={"test": True},
                    embedding=[0.4] * 1536,
                    source_id="test.com"
                )
            ]
            
            await provider.add_documents(test_docs)
            
            # Test hybrid search
            results = await provider.hybrid_search_documents(
                query_embedding=[0.45] * 1536,
                keyword_query="Python programming",
                match_count=1
            )
            
            assert len(results) >= 1
            assert "Python" in results[0].content

@pytest.mark.asyncio
class TestSupabaseProvider(DatabaseProviderTestSuite):
    """Test Supabase provider"""
    
    @pytest.fixture
    def provider_config(self):
        return {
            "type": DatabaseProvider.SUPABASE,
            "config": {
                "url": os.getenv("TEST_SUPABASE_URL"),
                "service_key": os.getenv("TEST_SUPABASE_SERVICE_KEY")
            }
        }

@pytest.mark.asyncio  
class TestSQLiteProvider(DatabaseProviderTestSuite):
    """Test SQLite provider"""
    
    @pytest.fixture
    def provider_config(self, tmp_path):
        return {
            "type": DatabaseProvider.SQLITE,
            "config": {
                "db_path": str(tmp_path / "test.sqlite"),
                "embedding_dimension": 1536
            }
        }

# Add similar test classes for other providers...
```

### 5.2 Integration Tests

**File**: `tests/test_integration.py`

```python
import pytest
import json
from src.crawl4ai_mcp import mcp
from src.database.config import DatabaseConfig

@pytest.mark.asyncio
async def test_crawl_and_search_workflow():
    """Test the complete crawl and search workflow"""
    
    # This would require a test MCP client
    # and mock websites for crawling
    
    # 1. Test crawl_single_page
    crawl_result = await mcp.call_tool(
        "crawl_single_page",
        {"url": "https://example.com/test"}
    )
    
    assert json.loads(crawl_result)["success"] == True
    
    # 2. Test get_available_sources
    sources_result = await mcp.call_tool("get_available_sources", {})
    sources = json.loads(sources_result)
    
    assert sources["success"] == True
    assert len(sources["sources"]) > 0
    
    # 3. Test perform_rag_query
    search_result = await mcp.call_tool(
        "perform_rag_query",
        {"query": "test query", "match_count": 5}
    )
    
    search_data = json.loads(search_result)
    assert search_data["success"] == True

@pytest.mark.asyncio
async def test_provider_switching():
    """Test switching between different database providers"""
    
    # Test with SQLite
    os.environ["VECTOR_DB_PROVIDER"] = "sqlite"
    config = DatabaseConfig.from_env()
    assert config.provider == DatabaseProvider.SQLITE
    
    # Test with Supabase  
    os.environ["VECTOR_DB_PROVIDER"] = "supabase"
    config = DatabaseConfig.from_env()
    assert config.provider == DatabaseProvider.SUPABASE
```

### 5.3 Performance Tests

**File**: `tests/test_performance.py`

```python
import pytest
import time
import asyncio
from src.database.base import DocumentChunk

@pytest.mark.asyncio
async def test_batch_insertion_performance(provider):
    """Test performance of batch document insertion"""
    
    # Create 100 test documents
    documents = []
    for i in range(100):
        doc = DocumentChunk(
            url=f"https://test.com/perf{i}",
            chunk_number=0,
            content=f"Performance test document {i} with some content to search",
            metadata={"test": True, "index": i},
            embedding=[0.1 + i * 0.001] * 1536,
            source_id="test.com"
        )
        documents.append(doc)
    
    # Measure insertion time
    start_time = time.time()
    await provider.add_documents(documents, batch_size=20)
    insertion_time = time.time() - start_time
    
    # Should complete within reasonable time (adjust threshold as needed)
    assert insertion_time < 30.0  # 30 seconds for 100 documents
    
    print(f"Inserted 100 documents in {insertion_time:.2f} seconds")

@pytest.mark.asyncio
async def test_search_performance(provider):
    """Test search performance with large dataset"""
    
    # First add documents (reuse from previous test)
    # Then test search performance
    
    start_time = time.time()
    results = await provider.search_documents(
        query_embedding=[0.5] * 1536,
        match_count=10
    )
    search_time = time.time() - start_time
    
    # Search should be fast
    assert search_time < 5.0  # 5 seconds max for search
    assert len(results) <= 10
    
    print(f"Search completed in {search_time:.3f} seconds")
```

## Implementation Timeline Summary

### Week 1-2: Foundation
- [ ] Create database abstraction layer interfaces
- [ ] Implement configuration management
- [ ] Create provider factory
- [ ] Migrate Supabase provider to new architecture
- [ ] Update dependency injection in main application

### Week 3: Additional Providers
- [ ] Implement Pinecone provider
- [ ] Implement SQLite provider  
- [ ] Basic Weaviate provider implementation
- [ ] Update build and dependency configurations

### Week 4: Integration and Testing
- [ ] Complete provider implementations
- [ ] Create comprehensive test suite
- [ ] Update Docker configurations
- [ ] Performance optimization and benchmarking

### Week 5: Documentation and Deployment
- [ ] Update documentation
- [ ] Create migration guides
- [ ] Deploy and test different configurations
- [ ] Performance tuning and optimization

## Migration from Current System

### For Existing Users

1. **Backup existing data** from Supabase
2. **Update environment variables** to include `VECTOR_DB_PROVIDER=supabase`
3. **Upgrade to new version** - backward compatibility maintained
4. **Test functionality** with existing data
5. **Optionally migrate** to other providers using provided tools

### For New Deployments

1. **Choose provider** based on requirements:
   - **SQLite**: Local development, no dependencies
   - **Supabase**: Current production setup, SQL familiarity
   - **Pinecone**: High performance, managed service
   - **Weaviate**: Enterprise features, ML capabilities
2. **Set environment variables** for chosen provider
3. **Deploy using Docker** or direct Python installation
4. **Configure MCP client** to connect to server

This implementation plan provides a complete roadmap for creating a database-agnostic version of the mcp-crawl4ai-rag system while maintaining all existing functionality and enabling support for multiple vector database backends.