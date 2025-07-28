# [STORY-1] Configurable Embedding Dimensions & Ollama Embedding Provider

## Story Overview
**As a** developer using the MCP Crawl4AI RAG system  
**I want** configurable embedding dimensions and Ollama embedding support  
**So that** I can use local embedding models with different dimensions while maintaining data integrity

## Acceptance Criteria

### AC1: Configurable Embedding Dimensions
- [ ] Database providers support configurable embedding dimensions
- [ ] Embedding dimension is configurable via environment variable `EMBEDDING_DIMENSION`
- [ ] Default dimension remains 1536 for backward compatibility with OpenAI
- [ ] Database initialization validates embedding dimension configuration
- [ ] Existing data with different dimensions is detected and handled gracefully

### AC2: Ollama Embedding Client Implementation
- [ ] New `OllamaEmbeddingProvider` class implemented following existing patterns
- [ ] Ollama client connects to configurable endpoint (default: `http://localhost:11434`)
- [ ] Embedding model configurable via `OLLAMA_EMBEDDING_MODEL` environment variable
- [ ] Automatic dimension detection from Ollama model metadata
- [ ] Batch embedding support for performance optimization
- [ ] Error handling with retry logic similar to OpenAI implementation

### AC3: AI Provider Factory Pattern
- [ ] New `AIProviderFactory` similar to existing `DatabaseProviderFactory`
- [ ] Environment variable `AI_PROVIDER` controls selection (openai|ollama)
- [ ] Provider interface abstracts embedding generation
- [ ] Consistent API across OpenAI and Ollama providers
- [ ] Provider-specific configuration validation

### AC4: Database Integration
- [ ] All database providers updated to handle variable embedding dimensions
- [ ] SQLite provider creates tables with configurable vector size
- [ ] Supabase provider validates pgvector dimension configuration
- [ ] Neo4j provider handles variable-length vector properties
- [ ] Pinecone provider validates index dimension compatibility

### AC5: Dimension Validation & Safety
- [ ] Startup validation checks embedding dimension compatibility
- [ ] Clear error messages for dimension mismatches
- [ ] Warning when changing embedding dimensions on existing data
- [ ] Migration guidance provided for dimension changes
- [ ] Database operations reject embeddings with wrong dimensions

## Technical Implementation Notes

### Files to Modify/Create:
- `src/ai/` (new directory)
  - `__init__.py`
  - `base.py` - Abstract embedding provider interface
  - `factory.py` - AI provider factory
  - `openai_provider.py` - Refactored OpenAI implementation
  - `ollama_provider.py` - New Ollama implementation

### Database Provider Updates:
- `src/database/base.py` - Add embedding_dimension to configuration
- `src/database/config.py` - Add dimension validation
- `src/database/providers/sqlite_provider.py` - Dynamic vector column size
- `src/database/providers/supabase_provider.py` - Dimension validation
- `src/database/providers/neo4j_provider.py` - Variable vector properties
- `src/database/providers/pinecone_provider.py` - Index dimension checks

### Configuration Updates:
- Add `AI_PROVIDER` environment variable
- Add `OLLAMA_ENDPOINT` environment variable (default: http://localhost:11434)
- Add `OLLAMA_EMBEDDING_MODEL` environment variable
- Add `EMBEDDING_DIMENSION` environment variable (auto-detect or manual)

### Key Implementation Details:

#### Ollama Client Integration
```python
class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: Dict[str, Any]):
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.model = config["model"]
        self.dimension = self._detect_dimension()
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding generation with Ollama API
        # Error handling and retry logic
        # Dimension validation
```

#### Dimension Validation
```python
def validate_embedding_dimension(stored_dim: int, new_dim: int) -> None:
    if stored_dim != new_dim:
        raise DimensionMismatchError(
            f"Embedding dimension mismatch: stored={stored_dim}, new={new_dim}. "
            f"To change dimensions, migrate existing data or use new database."
        )
```

## Testing Requirements

### Unit Tests
- [ ] Ollama embedding provider with mocked API calls
- [ ] Dimension validation logic
- [ ] AI provider factory selection
- [ ] Database provider dimension handling

### Integration Tests  
- [ ] Ollama embedding generation (with local Ollama instance)
- [ ] Dimension compatibility across all database providers
- [ ] Provider switching with same dimensions
- [ ] Error handling for dimension mismatches

### Compatibility Tests
- [ ] Existing OpenAI functionality unchanged
- [ ] All database providers work with configurable dimensions
- [ ] Migration from fixed to configurable dimensions
- [ ] Rollback to OpenAI provider

## Dependencies
- `ollama` Python client library
- Local Ollama installation for testing
- Environment variable configuration updates

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Unit and integration tests pass
- [ ] Existing OpenAI functionality preserved
- [ ] Documentation updated with configuration options
- [ ] Code review completed
- [ ] Dimension validation prevents data corruption
- [ ] Performance comparable to existing OpenAI implementation

## Risk Mitigation
- **Risk**: Breaking existing OpenAI embeddings
- **Mitigation**: Default to OpenAI provider, extensive compatibility testing
- **Risk**: Dimension mismatches corrupting vector searches  
- **Mitigation**: Strict validation, clear error messages, migration guidance