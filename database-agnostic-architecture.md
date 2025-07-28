# Database-Agnostic mcp-crawl4ai-rag Architecture

## Executive Summary

This document outlines the architectural design for transforming the current mcp-crawl4ai-rag system from a Supabase-specific implementation into a database-agnostic platform that supports multiple vector database backends while maintaining all existing RAG capabilities and MCP server functionality.

## Current System Analysis

### Architecture Overview
The current system is built around:
- **MCP Server**: FastMCP-based server providing web crawling and RAG tools
- **Database Layer**: Hardcoded Supabase integration with pgvector for vector storage
- **RAG Strategies**: Contextual embeddings, hybrid search, agentic RAG, reranking
- **Knowledge Graph**: Neo4j integration for AI hallucination detection
- **Crawling Engine**: Crawl4AI with parallel processing and memory-adaptive dispatching

### Key Components
1. **Core MCP Tools**: `crawl_single_page`, `smart_crawl_url`, `perform_rag_query`, `get_available_sources`
2. **Specialized Tools**: `search_code_examples`, knowledge graph tools
3. **Database Operations**: Document storage, vector search, code example management
4. **Embedding Pipeline**: OpenAI embeddings with contextual enhancement

## Database-Agnostic Architecture Design

### 1. Database Abstraction Layer

#### 1.1 Core Interfaces

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VectorSearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    url: str
    chunk_number: int

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
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
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
    async def add_code_examples(self, examples: List[CodeExample], batch_size: int = 20) -> None:
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
    async def get_sources(self) -> List[Dict[str, Any]]:
        """Get all available sources"""
        pass
    
    @abstractmethod
    async def update_source_info(self, source_id: str, summary: str, word_count: int) -> None:
        """Update source information"""
        pass
    
    @abstractmethod
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        pass
    
    @abstractmethod
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        pass
```

#### 1.2 Provider Factory

```python
from enum import Enum

class DatabaseProvider(Enum):
    SUPABASE = "supabase"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"
    NEO4J_VECTOR = "neo4j_vector"

class DatabaseProviderFactory:
    """Factory for creating database provider instances"""
    
    @staticmethod
    def create_provider(provider_type: DatabaseProvider, config: Dict[str, Any]) -> VectorDatabaseProvider:
        if provider_type == DatabaseProvider.SUPABASE:
            return SupabaseProvider(config)
        elif provider_type == DatabaseProvider.PINECONE:
            return PineconeProvider(config)
        elif provider_type == DatabaseProvider.WEAVIATE:
            return WeaviateProvider(config)
        elif provider_type == DatabaseProvider.SQLITE:
            return SQLiteProvider(config)
        elif provider_type == DatabaseProvider.NEO4J_VECTOR:
            return Neo4jVectorProvider(config)
        else:
            raise ValueError(f"Unsupported database provider: {provider_type}")
```

### 2. Database Provider Implementations

#### 2.1 Supabase Provider (Current Implementation)

```python
class SupabaseProvider(VectorDatabaseProvider):
    """Supabase/pgvector implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
    
    async def initialize(self) -> None:
        from supabase import create_client
        self.client = create_client(
            self.config["url"],
            self.config["service_key"]
        )
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        # Implementation using existing Supabase logic
        pass
    
    async def search_documents(self, query_embedding: List[float], match_count: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        # Implementation using match_crawled_pages RPC
        pass
```

#### 2.2 Pinecone Provider

```python
class PineconeProvider(VectorDatabaseProvider):
    """Pinecone vector database implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index = None
    
    async def initialize(self) -> None:
        import pinecone
        pinecone.init(
            api_key=self.config["api_key"],
            environment=self.config["environment"]
        )
        self.index = pinecone.Index(self.config["index_name"])
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        vectors = []
        for doc in documents:
            vectors.append({
                "id": f"{doc.url}#{doc.chunk_number}",
                "values": doc.embedding,
                "metadata": {
                    "url": doc.url,
                    "content": doc.content,
                    "source_id": doc.source_id,
                    **doc.metadata
                }
            })
        self.index.upsert(vectors)
```

#### 2.3 Weaviate Provider

```python
class WeaviateProvider(VectorDatabaseProvider):
    """Weaviate vector database implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
    
    async def initialize(self) -> None:
        import weaviate
        self.client = weaviate.Client(
            url=self.config["url"],
            auth_client_secret=weaviate.AuthApiKey(api_key=self.config["api_key"])
        )
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        with self.client.batch as batch:
            for doc in documents:
                batch.add_data_object({
                    "url": doc.url,
                    "content": doc.content,
                    "source_id": doc.source_id,
                    "metadata": doc.metadata
                }, "Document", vector=doc.embedding)
```

#### 2.4 SQLite Provider (Local)

```python
class SQLiteProvider(VectorDatabaseProvider):
    """Local SQLite with vector similarity using sentence-transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("db_path", "./vector_db.sqlite")
        self.connection = None
    
    async def initialize(self) -> None:
        import sqlite3
        import numpy as np
        
        self.connection = sqlite3.connect(self.db_path)
        await self._create_tables()
    
    async def search_documents(self, query_embedding: List[float], match_count: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load all embeddings and compute similarity
        cursor = self.connection.cursor()
        cursor.execute("SELECT id, url, content, metadata, embedding FROM documents")
        results = []
        
        for row in cursor.fetchall():
            doc_embedding = np.frombuffer(row[4], dtype=np.float32)
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            results.append(VectorSearchResult(
                id=row[0],
                url=row[1],
                content=row[2],
                metadata=json.loads(row[3]),
                similarity=similarity,
                chunk_number=0
            ))
        
        return sorted(results, key=lambda x: x.similarity, reverse=True)[:match_count]
```

#### 2.5 Neo4j Vector Provider

```python
class Neo4jVectorProvider(VectorDatabaseProvider):
    """Neo4j vector index implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
    
    async def initialize(self) -> None:
        from neo4j import AsyncGraphDatabase
        self.driver = AsyncGraphDatabase.driver(
            self.config["uri"],
            auth=(self.config["user"], self.config["password"])
        )
        await self._create_vector_indexes()
    
    async def search_documents(self, query_embedding: List[float], match_count: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        async with self.driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('document_embeddings', $limit, $query_vector)
            YIELD node, score
            RETURN node.id as id, node.url as url, node.content as content, 
                   node.metadata as metadata, score
            """
            result = await session.run(query, query_vector=query_embedding, limit=match_count)
            return [VectorSearchResult(
                id=record["id"],
                url=record["url"],
                content=record["content"],
                metadata=record["metadata"],
                similarity=record["score"],
                chunk_number=0
            ) async for record in result]
```

### 3. Configuration Management

#### 3.1 Database Configuration

```python
@dataclass
class DatabaseConfig:
    provider: DatabaseProvider
    config: Dict[str, Any]
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        provider_name = os.getenv("VECTOR_DB_PROVIDER", "supabase").lower()
        provider = DatabaseProvider(provider_name)
        
        if provider == DatabaseProvider.SUPABASE:
            config = {
                "url": os.getenv("SUPABASE_URL"),
                "service_key": os.getenv("SUPABASE_SERVICE_KEY")
            }
        elif provider == DatabaseProvider.PINECONE:
            config = {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "environment": os.getenv("PINECONE_ENVIRONMENT"),
                "index_name": os.getenv("PINECONE_INDEX_NAME", "crawl4ai-rag")
            }
        elif provider == DatabaseProvider.WEAVIATE:
            config = {
                "url": os.getenv("WEAVIATE_URL"),
                "api_key": os.getenv("WEAVIATE_API_KEY")
            }
        elif provider == DatabaseProvider.SQLITE:
            config = {
                "db_path": os.getenv("SQLITE_DB_PATH", "./vector_db.sqlite")
            }
        elif provider == DatabaseProvider.NEO4J_VECTOR:
            config = {
                "uri": os.getenv("NEO4J_URI"),
                "user": os.getenv("NEO4J_USER"),
                "password": os.getenv("NEO4J_PASSWORD")
            }
        
        return cls(provider=provider, config=config)
```

#### 3.2 Environment Variables

```bash
# Database Provider Selection
VECTOR_DB_PROVIDER=supabase  # supabase, pinecone, weaviate, sqlite, neo4j_vector

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=crawl4ai-rag

# Weaviate Configuration
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_api_key

# SQLite Configuration (Local)
SQLITE_DB_PATH=./vector_db.sqlite

# Neo4j Vector Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

### 4. Dependency Injection Integration

#### 4.1 Updated Context

```python
@dataclass
class Crawl4AIContext:
    """Enhanced context with database abstraction"""
    crawler: AsyncWebCrawler
    vector_db: VectorDatabaseProvider
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None
    repo_extractor: Optional[Any] = None

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """Updated lifespan with database provider injection"""
    
    # Initialize crawler
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize database provider
    db_config = DatabaseConfig.from_env()
    vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
    await vector_db.initialize()
    
    # Initialize other components...
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
    
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
```

#### 4.2 Updated Tool Implementation

```python
@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """Updated RAG query using database abstraction"""
    try:
        vector_db = ctx.request_context.lifespan_context.vector_db
        
        # Create query embedding
        query_embedding = create_embedding(query)
        
        # Prepare filter
        filter_metadata = {"source": source} if source and source.strip() else None
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        if use_hybrid_search and hasattr(vector_db, 'hybrid_search_documents'):
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
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(
                ctx.request_context.lifespan_context.reranking_model, 
                query, 
                results, 
                content_key="content"
            )
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": use_reranking,
            "results": [result.__dict__ for result in results],
            "count": len(results)
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)
```

### 5. Migration Strategy

#### 5.1 Phase 1: Database Abstraction Layer
1. Create abstract interfaces and base classes
2. Implement Supabase provider maintaining existing functionality
3. Update dependency injection and context management
4. Maintain backward compatibility

#### 5.2 Phase 2: Additional Providers
1. Implement Pinecone provider for cloud vector search
2. Implement Weaviate provider for enterprise deployments
3. Implement SQLite provider for local development
4. Add Neo4j vector provider for graph-enhanced RAG

#### 5.3 Phase 3: Enhanced Features
1. Add provider-specific optimizations
2. Implement cross-provider migration tools
3. Add performance benchmarking
4. Enhanced monitoring and logging

### 6. Deployment Configurations

#### 6.1 Docker Compose with Multiple Providers

```yaml
version: '3.8'
services:
  # Supabase deployment
  mcp-crawl4ai-supabase:
    build: .
    environment:
      - VECTOR_DB_PROVIDER=supabase
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
    ports:
      - "8051:8051"
  
  # Local SQLite deployment
  mcp-crawl4ai-sqlite:
    build: .
    environment:
      - VECTOR_DB_PROVIDER=sqlite
      - SQLITE_DB_PATH=/data/vector_db.sqlite
    volumes:
      - ./data:/data
    ports:
      - "8052:8051"
  
  # Weaviate deployment
  mcp-crawl4ai-weaviate:
    build: .
    environment:
      - VECTOR_DB_PROVIDER=weaviate
      - WEAVIATE_URL=http://weaviate:8080
    depends_on:
      - weaviate
    ports:
      - "8053:8051"
  
  weaviate:
    image: semitechnologies/weaviate:latest
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    ports:
      - "8080:8080"
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

### 7. Performance Considerations

#### 7.1 Provider-Specific Optimizations

| Provider | Strengths | Optimizations |
|----------|-----------|---------------|
| Supabase | SQL familiarity, hybrid search | Connection pooling, RPC functions |
| Pinecone | High performance, managed | Batch operations, namespace usage |
| Weaviate | Schema flexibility, ML features | Batch imports, vector caching |
| SQLite | Local development, no dependencies | In-memory indexes, WAL mode |
| Neo4j | Graph relationships, complex queries | Vector indexes, relationship caching |

#### 7.2 Async Operations and Batch Processing

```python
class BatchProcessor:
    """Optimized batch processing for different providers"""
    
    def __init__(self, provider: VectorDatabaseProvider):
        self.provider = provider
        self.batch_sizes = {
            DatabaseProvider.SUPABASE: 20,
            DatabaseProvider.PINECONE: 100,
            DatabaseProvider.WEAVIATE: 50,
            DatabaseProvider.SQLITE: 10,
            DatabaseProvider.NEO4J_VECTOR: 25
        }
    
    async def process_documents(self, documents: List[DocumentChunk]) -> None:
        batch_size = self.batch_sizes.get(type(self.provider), 20)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await self.provider.add_documents(batch, batch_size)
            
            # Provider-specific optimizations
            if isinstance(self.provider, PineconeProvider):
                await asyncio.sleep(0.1)  # Rate limiting
            elif isinstance(self.provider, SQLiteProvider):
                await self.provider.optimize_indexes()
```

### 8. Testing Strategy

#### 8.1 Provider Test Suite

```python
import pytest
from abc import ABC

class DatabaseProviderTestSuite(ABC):
    """Abstract test suite for all database providers"""
    
    @pytest.fixture
    def provider(self) -> VectorDatabaseProvider:
        raise NotImplementedError
    
    async def test_document_operations(self, provider):
        # Test document CRUD operations
        documents = [create_test_document()]
        await provider.add_documents(documents)
        
        results = await provider.search_documents(test_embedding, match_count=1)
        assert len(results) == 1
        assert results[0].content == documents[0].content
    
    async def test_code_example_operations(self, provider):
        # Test code example operations
        examples = [create_test_code_example()]
        await provider.add_code_examples(examples)
        
        results = await provider.search_code_examples(test_embedding, match_count=1)
        assert len(results) == 1
    
    async def test_hybrid_search(self, provider):
        if hasattr(provider, 'hybrid_search_documents'):
            results = await provider.hybrid_search_documents(
                test_embedding, "test query", match_count=5
            )
            assert isinstance(results, list)

class TestSupabaseProvider(DatabaseProviderTestSuite):
    @pytest.fixture
    def provider(self):
        return SupabaseProvider(test_config)

class TestPineconeProvider(DatabaseProviderTestSuite):
    @pytest.fixture
    def provider(self):
        return PineconeProvider(test_config)
```

### 9. Monitoring and Observability

#### 9.1 Provider Metrics

```python
import time
from typing import Dict, Any
import logging

class DatabaseMetrics:
    """Metrics collection for database operations"""
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
    
    def record_operation(self, operation: str, duration: float, success: bool):
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
            self.error_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        if not success:
            self.error_counts[operation] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for operation in self.operation_times:
            times = self.operation_times[operation]
            stats[operation] = {
                "count": self.operation_counts[operation],
                "errors": self.error_counts[operation],
                "avg_time": sum(times) / len(times) if times else 0,
                "total_time": sum(times)
            }
        return stats

def monitor_database_operation(operation_name: str):
    """Decorator for monitoring database operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logging.error(f"Database operation {operation_name} failed: {e}")
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_operation(operation_name, duration, success)
        return wrapper
    return decorator
```

## Conclusion

This database-agnostic architecture provides:

1. **Flexibility**: Support for multiple vector database backends
2. **Maintainability**: Clean separation of concerns through abstraction
3. **Scalability**: Provider-specific optimizations and async operations
4. **Testability**: Comprehensive test suite for all providers
5. **Observability**: Built-in metrics and monitoring
6. **Migration Path**: Smooth transition from current Supabase implementation

The design maintains all existing RAG strategies while enabling deployment across different infrastructure requirements, from local development with SQLite to enterprise deployments with Weaviate or cloud-native solutions with Pinecone.