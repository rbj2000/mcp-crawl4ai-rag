# Technical Specification: Database-Agnostic mcp-crawl4ai-rag

## Document Information

**Document Type**: Technical Specification  
**Project**: Database-Agnostic mcp-crawl4ai-rag System  
**Version**: 1.0  
**Date**: 2025-01-27  
**Status**: Implementation Ready

## Executive Summary

This technical specification defines the requirements and architecture for transforming the current mcp-crawl4ai-rag system from a Supabase-specific implementation into a database-agnostic platform supporting multiple vector database backends while maintaining all existing RAG capabilities.

## 1. System Requirements

### 1.1 Functional Requirements

#### FR-1: Database Provider Abstraction
- **Requirement**: The system must support multiple vector database providers through a unified interface
- **Providers**: Supabase/pgvector, Pinecone, Weaviate, SQLite, Neo4j Vector
- **Interface**: Common API for CRUD operations, vector search, and metadata management
- **Priority**: Critical

#### FR-2: Backward Compatibility
- **Requirement**: Existing Supabase deployments must continue to function without modification
- **Migration Path**: Seamless upgrade path from current version
- **Data Preservation**: No data loss during migration
- **Priority**: Critical

#### FR-3: RAG Strategy Preservation
- **Requirement**: All existing RAG strategies must be maintained across all providers
- **Strategies**: Contextual embeddings, hybrid search, agentic RAG, reranking, knowledge graphs
- **Performance**: No degradation in search quality or performance
- **Priority**: Critical

#### FR-4: MCP Server Functionality
- **Requirement**: All existing MCP tools must function with any database provider
- **Tools**: crawl_single_page, smart_crawl_url, perform_rag_query, search_code_examples, etc.
- **API Compatibility**: No changes to MCP tool interfaces
- **Priority**: Critical

#### FR-5: Configuration Management
- **Requirement**: Provider selection and configuration through environment variables
- **Dynamic Switching**: Ability to change providers without code changes
- **Validation**: Configuration validation and error reporting
- **Priority**: High

### 1.2 Non-Functional Requirements

#### NFR-1: Performance
- **Vector Search Latency**: < 500ms for 1000 document corpus
- **Batch Insert Performance**: > 100 documents/second
- **Memory Usage**: < 2GB RAM for standard deployment
- **Concurrent Users**: Support 10+ concurrent search requests
- **Priority**: High

#### NFR-2: Scalability
- **Document Capacity**: Support 100K+ documents per source
- **Provider Scaling**: Leverage provider-specific scaling capabilities
- **Horizontal Scaling**: Support multiple server instances
- **Priority**: High

#### NFR-3: Reliability
- **Uptime**: 99.9% availability target
- **Error Handling**: Graceful degradation on provider failures
- **Data Integrity**: ACID compliance where supported
- **Backup/Recovery**: Support for data backup and restore
- **Priority**: High

#### NFR-4: Security
- **Authentication**: Secure provider authentication
- **Data Encryption**: Encryption in transit and at rest
- **API Security**: Secure MCP protocol implementation
- **Access Control**: Provider-level access controls
- **Priority**: Medium

#### NFR-5: Maintainability
- **Code Quality**: Clean architecture with separation of concerns
- **Documentation**: Comprehensive API and deployment documentation
- **Testing**: 90%+ test coverage across all providers
- **Monitoring**: Built-in metrics and health checks
- **Priority**: Medium

## 2. Architecture Specification

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Client Layer                            │
│  (Claude Desktop, Windsurf, Claude Code, etc.)                 │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ MCP Protocol
┌─────────────────────────────────────┴───────────────────────────┐
│                   FastMCP Server                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MCP Tools Layer                            │   │
│  │  crawl_single_page  │  perform_rag_query               │   │
│  │  smart_crawl_url    │  search_code_examples            │   │
│  │  get_available_sources  │  knowledge_graph_tools       │   │
│  └─────────────────┬───────────────────────────────────────┘   │
└────────────────────┼───────────────────────────────────────────┘
                     │ Abstraction Layer API
┌────────────────────┴───────────────────────────────────────────┐
│              Database Abstraction Layer                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           VectorDatabaseProvider Interface              │   │
│  │  - add_documents()      - search_documents()           │   │
│  │  - add_code_examples()  - search_code_examples()       │   │
│  │  - get_sources()        - hybrid_search()              │   │
│  └─────────────────┬───────────────────────────────────────┘   │
│  ┌─────────────────┴───────────────────────────────────────┐   │
│  │              Provider Factory                           │   │
│  │  DatabaseProvider.SUPABASE → SupabaseProvider          │   │
│  │  DatabaseProvider.PINECONE → PineconeProvider          │   │
│  │  DatabaseProvider.WEAVIATE → WeaviateProvider          │   │
│  │  DatabaseProvider.SQLITE   → SQLiteProvider            │   │
│  │  DatabaseProvider.NEO4J    → Neo4jVectorProvider       │   │
│  └─────────────────┬───────────────────────────────────────┘   │
└────────────────────┼───────────────────────────────────────────┘
                     │ Provider-Specific APIs
┌────────────────────┴───────────────────────────────────────────┐
│                Provider Implementations                         │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│ │  Supabase   │ │  Pinecone   │ │  Weaviate   │ │   SQLite    ││
│ │ Provider    │ │ Provider    │ │ Provider    │ │ Provider    ││
│ │             │ │             │ │             │ │             ││
│ │ pgvector    │ │ Pinecone    │ │ Weaviate    │ │ Local DB    ││
│ │ SQL RPC     │ │ REST API    │ │ GraphQL     │ │ File-based  ││
│ │ Hybrid      │ │ Namespace   │ │ Multi-Class │ │ Similarity  ││
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Model Specification

#### 2.2.1 Core Data Types

```python
@dataclass
class VectorSearchResult:
    id: str                    # Unique identifier
    content: str               # Document content
    metadata: Dict[str, Any]   # Arbitrary metadata
    similarity: float          # Similarity score (0.0-1.0)
    url: str                   # Source URL
    chunk_number: int          # Chunk index
    source_id: str             # Source identifier

@dataclass  
class DocumentChunk:
    url: str                   # Source URL
    chunk_number: int          # Chunk index within document
    content: str               # Chunk content
    metadata: Dict[str, Any]   # Chunk metadata
    embedding: List[float]     # Vector embedding (1536 dimensions)
    source_id: str             # Source domain/identifier

@dataclass
class CodeExample:
    url: str                   # Source URL
    chunk_number: int          # Example index
    content: str               # Code content
    summary: str               # AI-generated summary
    metadata: Dict[str, Any]   # Example metadata
    embedding: List[float]     # Vector embedding
    source_id: str             # Source identifier

@dataclass
class SourceInfo:
    source_id: str             # Unique source identifier
    summary: str               # Source description
    total_words: int           # Total word count
    created_at: str            # Creation timestamp
    updated_at: str            # Last update timestamp
```

#### 2.2.2 Database Schema Mapping

| Entity | Supabase | Pinecone | Weaviate | SQLite | Neo4j |
|--------|----------|----------|----------|---------|-------|
| Documents | crawled_pages table | Index vectors | Document class | documents table | Document nodes |
| Code Examples | code_examples table | Separate index | CodeExample class | code_examples table | Code nodes |
| Sources | sources table | Metadata namespace | Source class | sources table | Source nodes |
| Embeddings | vector column | Vector values | Vector property | blob column | Vector index |

### 2.3 Interface Specifications

#### 2.3.1 VectorDatabaseProvider Interface

```python
class VectorDatabaseProvider(ABC):
    """Abstract interface for all vector database providers"""
    
    # Lifecycle Management
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database connection and schema"""
    
    @abstractmethod
    async def close(self) -> None:
        """Close database connections"""
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Verify database health and connectivity"""
    
    # Document Operations
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        batch_size: int = 20
    ) -> None:
        """Batch insert document chunks"""
    
    @abstractmethod
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Vector similarity search for documents"""
    
    # Code Example Operations  
    @abstractmethod
    async def add_code_examples(
        self, 
        examples: List[CodeExample], 
        batch_size: int = 20
    ) -> None:
        """Batch insert code examples"""
    
    @abstractmethod
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Vector similarity search for code examples"""
    
    # Source Management
    @abstractmethod
    async def get_sources(self) -> List[SourceInfo]:
        """Retrieve all available sources"""
    
    @abstractmethod
    async def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int
    ) -> None:
        """Create or update source information"""
    
    # Maintenance Operations
    @abstractmethod
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete all data associated with URLs"""
    
    # Optional Advanced Features
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Hybrid search combining vector and keyword search"""
        # Default implementation falls back to vector-only search
        return await self.search_documents(
            query_embedding, match_count, filter_metadata
        )
```

#### 2.3.2 Configuration Interface

```python
@dataclass
class DatabaseConfig:
    provider: DatabaseProvider     # Provider type enum
    config: Dict[str, Any]        # Provider-specific configuration
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load configuration from environment variables"""
    
    def validate(self) -> None:
        """Validate configuration completeness"""
    
    def get_connection_string(self) -> str:
        """Generate provider-specific connection string"""
```

### 2.4 Provider-Specific Implementations

#### 2.4.1 Supabase Provider

**Features**:
- Native SQL and pgvector support
- RPC functions for optimized vector search
- Built-in hybrid search using ILIKE + vector similarity
- Connection pooling and transaction support

**Configuration**:
```python
{
    "url": "https://project.supabase.co",
    "service_key": "eyJ...",
    "schema": "public"  # optional
}
```

**Optimizations**:
- Batch insertions with conflict resolution
- RPC functions for complex queries
- Connection pooling
- Prepared statements for repeated queries

#### 2.4.2 Pinecone Provider

**Features**:
- Managed vector database service
- Automatic scaling and performance optimization
- Namespace support for multi-tenancy
- Built-in filtering and metadata support

**Configuration**:
```python
{
    "api_key": "...",
    "environment": "us-west1-gcp",
    "index_name": "crawl4ai-rag",
    "namespace": "default"  # optional
}
```

**Optimizations**:
- Batch upserts up to 100 vectors
- Efficient metadata filtering
- Namespace isolation
- Async operations with rate limiting

#### 2.4.3 Weaviate Provider

**Features**:
- Schema-based vector database
- Built-in ML capabilities
- GraphQL query interface
- Multi-modal support

**Configuration**:
```python
{
    "url": "http://localhost:8080",
    "api_key": "...",  # optional
    "auth_config": {...},  # optional
    "timeout": 60
}
```

**Optimizations**:
- Batch imports with auto-schema
- Efficient where filters
- Vector caching
- Connection reuse

#### 2.4.4 SQLite Provider

**Features**:
- Local file-based database
- No external dependencies
- ACID compliance
- Full-text search capabilities

**Configuration**:
```python
{
    "db_path": "./vector_db.sqlite",
    "embedding_dimension": 1536,
    "enable_fts": true  # Full-text search
}
```

**Optimizations**:
- WAL mode for concurrent access
- In-memory indexes for embeddings
- Batch transactions
- FTS5 for keyword search

#### 2.4.5 Neo4j Vector Provider

**Features**:
- Graph database with vector indexes
- APOC and GDS library support
- Cypher query language
- Relationship-aware search

**Configuration**:
```python
{
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "...",
    "database": "neo4j",
    "vector_index": "document_embeddings"
}
```

**Optimizations**:
- Vector indexes for similarity search
- Relationship traversal for context
- Batch node creation
- Connection pooling

## 3. Technical Implementation Details

### 3.1 Embedding Pipeline

#### 3.1.1 Embedding Generation
- **Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Batch Processing**: Up to 100 texts per API call
- **Error Handling**: Exponential backoff with fallback to individual embeddings
- **Caching**: Optional embedding caching for repeated content

#### 3.1.2 Contextual Embeddings
- **Process**: Full document context + chunk content → LLM → enriched context → embedding
- **Model**: Configurable via MODEL_CHOICE environment variable
- **Parallelization**: ThreadPoolExecutor for concurrent LLM calls
- **Token Management**: Content truncation to stay within token limits

### 3.2 Search Algorithms

#### 3.2.1 Vector Similarity Search
- **Metric**: Cosine similarity (default for all providers)
- **Normalization**: L2 normalization of query vectors
- **Filtering**: Metadata-based pre/post filtering
- **Ranking**: Similarity score-based ranking

#### 3.2.2 Hybrid Search
- **Strategy**: Combine vector similarity + keyword search
- **Scoring**: Boost items appearing in both result sets
- **Fusion**: RRF (Reciprocal Rank Fusion) for result merging
- **Fallback**: Vector-only search if keyword search fails

#### 3.2.3 Reranking
- **Model**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Process**: Re-score top-k results with cross-encoder
- **Performance**: Local CPU inference, ~100-200ms overhead
- **Applicability**: Both document and code example search

### 3.3 Performance Optimizations

#### 3.3.1 Async Operations
- **Pattern**: Async/await throughout the stack
- **Concurrency**: Parallel document processing
- **Batching**: Provider-optimized batch sizes
- **Connection Pooling**: Reuse database connections

#### 3.3.2 Memory Management
- **Streaming**: Process large documents in chunks
- **Garbage Collection**: Explicit cleanup of large objects
- **Memory Monitoring**: Track memory usage and optimize
- **Lazy Loading**: Load embeddings on-demand

#### 3.3.3 Caching Strategy
- **Query Caching**: Cache frequent search results
- **Embedding Caching**: Cache embeddings for repeated content
- **Connection Caching**: Maintain persistent connections
- **TTL Management**: Time-based cache invalidation

### 3.4 Error Handling and Resilience

#### 3.4.1 Connection Management
- **Retry Logic**: Exponential backoff for transient failures
- **Circuit Breakers**: Fail fast on persistent failures
- **Health Checks**: Regular connectivity verification
- **Graceful Degradation**: Continue with reduced functionality

#### 3.4.2 Data Consistency
- **Atomic Operations**: Ensure data consistency across operations
- **Transaction Support**: Use transactions where available
- **Conflict Resolution**: Handle concurrent write conflicts
- **Backup Strategies**: Support for data backup and recovery

## 4. Deployment Specifications

### 4.1 Environment Configuration

#### 4.1.1 Required Environment Variables
```bash
# Core Configuration
VECTOR_DB_PROVIDER=supabase|pinecone|weaviate|sqlite|neo4j_vector
OPENAI_API_KEY=sk-...
MODEL_CHOICE=gpt-4o-mini

# Provider-Specific Configuration
# (See provider sections for details)

# RAG Strategy Configuration
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Performance Tuning
BATCH_SIZE=20
MAX_CONCURRENT=10
EMBEDDING_CACHE_TTL=3600
```

#### 4.1.2 Configuration Validation
- **Startup Validation**: Verify all required variables are set
- **Connection Testing**: Test database connectivity on startup
- **Schema Validation**: Ensure required tables/indexes exist
- **Permission Checking**: Verify necessary database permissions

### 4.2 Container Deployment

#### 4.2.1 Multi-Stage Docker Build
```dockerfile
# Base stage with common dependencies
FROM python:3.12-slim as base
RUN pip install uv

# Provider-specific stages
FROM base as supabase
RUN uv pip install supabase

FROM base as pinecone  
RUN uv pip install pinecone-client

FROM base as sqlite
RUN uv pip install nothing-additional

# Final stage selection based on build arg
FROM ${VECTOR_DB_PROVIDER:-supabase} as final
COPY src/ ./src/
CMD ["python", "src/crawl4ai_mcp.py"]
```

#### 4.2.2 Docker Compose Configurations
- **Development**: SQLite + local services
- **Production**: Provider-specific configurations
- **Testing**: Multi-provider test environments
- **Monitoring**: Integrated observability stack

### 4.3 Kubernetes Deployment

#### 4.3.1 Resource Requirements
```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

#### 4.3.2 Configuration Management
- **ConfigMaps**: Non-sensitive configuration
- **Secrets**: API keys and credentials
- **Environment**: Provider selection and tuning
- **Volume Mounts**: SQLite database persistence

### 4.4 Monitoring and Observability

#### 4.4.1 Metrics Collection
- **Performance Metrics**: Query latency, throughput, error rates
- **System Metrics**: Memory usage, CPU utilization, disk I/O
- **Business Metrics**: Document count, search volume, success rates
- **Provider Metrics**: Provider-specific performance indicators

#### 4.4.2 Logging Strategy
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Log Levels**: Debug, info, warning, error with appropriate filtering
- **Audit Logging**: Track data access and modifications
- **Error Tracking**: Comprehensive error capture and alerting

#### 4.4.3 Health Checks
- **Startup Probes**: Verify initialization completion
- **Liveness Probes**: Detect and recover from deadlocks
- **Readiness Probes**: Ensure service is ready to handle requests
- **Database Health**: Regular database connectivity checks

## 5. Migration and Compatibility

### 5.1 Migration Strategy

#### 5.1.1 Backward Compatibility
- **API Preservation**: No changes to MCP tool interfaces
- **Configuration Compatibility**: Default to Supabase for existing deployments
- **Data Format**: Maintain existing data structures and formats
- **Feature Parity**: All RAG strategies work across providers

#### 5.1.2 Migration Process
1. **Backup**: Export existing data from Supabase
2. **Upgrade**: Deploy new version with VECTOR_DB_PROVIDER=supabase
3. **Validate**: Verify all functionality works correctly
4. **Migrate**: Optionally switch to different provider
5. **Cleanup**: Remove old deployment once validated

#### 5.1.3 Data Export/Import
- **Export Tools**: Scripts to extract data from any provider
- **Import Tools**: Scripts to load data into any provider
- **Format Conversion**: Handle provider-specific data transformations
- **Validation**: Verify data integrity after migration

### 5.2 Version Compatibility

#### 5.2.1 API Versioning
- **Semantic Versioning**: Major.Minor.Patch version scheme
- **Breaking Changes**: Only in major version updates
- **Deprecation Policy**: 1 version deprecation notice
- **Compatibility Matrix**: Supported provider versions

#### 5.2.2 Database Schema Evolution
- **Schema Versioning**: Track schema versions per provider
- **Migration Scripts**: Automated schema updates
- **Rollback Support**: Ability to revert schema changes
- **Multi-Version Support**: Support for multiple schema versions

## 6. Testing Strategy

### 6.1 Unit Testing

#### 6.1.1 Provider Testing
- **Interface Compliance**: Verify all providers implement full interface
- **Functionality Testing**: Test CRUD operations for each provider
- **Error Handling**: Verify proper error propagation and handling
- **Performance Testing**: Benchmark operations across providers

#### 6.1.2 Integration Testing
- **End-to-End Workflows**: Test complete crawl-to-search workflows
- **Cross-Provider**: Verify functionality across different providers
- **MCP Integration**: Test tool interactions with various clients
- **Concurrency Testing**: Verify thread safety and concurrent access

### 6.2 Performance Testing

#### 6.2.1 Load Testing
- **Throughput**: Maximum requests per second
- **Latency**: 95th percentile response times
- **Scalability**: Performance with increasing data volumes
- **Resource Usage**: Memory and CPU utilization under load

#### 6.2.2 Stress Testing
- **Breaking Points**: Identify system limits
- **Recovery Testing**: Verify graceful degradation and recovery
- **Memory Stress**: Test behavior under memory pressure
- **Connection Limits**: Test maximum concurrent connections

### 6.3 Quality Assurance

#### 6.3.1 Code Quality
- **Coverage Target**: 90%+ test coverage across all providers
- **Code Review**: Mandatory peer review for all changes
- **Static Analysis**: Automated code quality checks
- **Documentation**: Comprehensive API and deployment documentation

#### 6.3.2 Security Testing
- **Vulnerability Scanning**: Regular dependency vulnerability scans
- **Penetration Testing**: API security verification
- **Access Control**: Verify proper authentication and authorization
- **Data Protection**: Ensure secure data handling practices

## 7. Success Criteria

### 7.1 Functional Success
- [ ] All five database providers (Supabase, Pinecone, Weaviate, SQLite, Neo4j) fully implemented
- [ ] All existing MCP tools function correctly with any provider
- [ ] All RAG strategies (contextual, hybrid, agentic, reranking, knowledge graph) work across providers
- [ ] Zero breaking changes to existing MCP tool interfaces
- [ ] Successful migration path from current Supabase-only implementation

### 7.2 Performance Success
- [ ] Vector search latency < 500ms for 1000 document corpus
- [ ] Batch insert performance > 100 documents/second
- [ ] Memory usage < 2GB RAM for standard deployment
- [ ] Support for 10+ concurrent search requests
- [ ] 99.9% uptime in production deployments

### 7.3 Quality Success
- [ ] 90%+ automated test coverage across all providers
- [ ] Comprehensive documentation for deployment and usage
- [ ] Clean architecture with clear separation of concerns
- [ ] Successful deployment across development, staging, and production environments
- [ ] Positive feedback from initial users across different providers

This technical specification provides the complete blueprint for implementing a database-agnostic version of the mcp-crawl4ai-rag system while maintaining all existing functionality and enabling support for multiple vector database backends.