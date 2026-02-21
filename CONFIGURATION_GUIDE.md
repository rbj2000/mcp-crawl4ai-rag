# AI Provider Configuration Guide

This guide provides comprehensive documentation for configuring the MCP Crawl4AI RAG system with multiple AI providers and deployment options.

## Configuration Overview

The system supports flexible configuration through environment variables, allowing you to:
- Choose between OpenAI, Ollama, vLLM, or hybrid AI providers
- Select from multiple vector database options
- Configure RAG strategies and performance settings
- Set up monitoring and debugging

## Environment Variable Reference

### AI Provider Configuration

#### Primary Provider Selection

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AI_PROVIDER` | `openai`, `ollama`, `vllm`, `mixed` | `openai` | Primary AI provider selection |
| `EMBEDDING_PROVIDER` | `openai`, `ollama`, `vllm` | Uses `AI_PROVIDER` | Provider for embedding generation |
| `LLM_PROVIDER` | `openai`, `ollama`, `vllm` | Uses `AI_PROVIDER` | Provider for LLM operations |

**Examples:**
```bash
# Single provider
AI_PROVIDER=openai

# vLLM provider (cloud-deployed)
AI_PROVIDER=vllm

# Mixed providers (cost optimization)
AI_PROVIDER=mixed
EMBEDDING_PROVIDER=vllm
LLM_PROVIDER=openai
```

#### OpenAI Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OPENAI_API_KEY` | - | Yes | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | No | Custom OpenAI endpoint |
| `OPENAI_ORGANIZATION` | - | No | OpenAI organization ID |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | No | Embedding model name |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | No | LLM model name |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | No | Embedding vector dimensions |
| `OPENAI_MAX_BATCH_SIZE` | `100` | No | Max texts per batch request |
| `OPENAI_MAX_RETRIES` | `3` | No | Retry attempts for failed requests |
| `OPENAI_TIMEOUT` | `30.0` | No | Request timeout in seconds |

**Example Configuration:**
```bash
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o
OPENAI_EMBEDDING_DIMENSIONS=3072
OPENAI_MAX_BATCH_SIZE=50
OPENAI_TIMEOUT=60.0
```

#### Ollama Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | No | Embedding model name |
| `OLLAMA_LLM_MODEL` | `llama3.2:1b` | No | LLM model name |
| `OLLAMA_EMBEDDING_DIMENSIONS` | `768` | No | Embedding vector dimensions |
| `OLLAMA_MAX_BATCH_SIZE` | `10` | No | Max texts per batch (Ollama is slower) |
| `OLLAMA_MAX_RETRIES` | `3` | No | Retry attempts for failed requests |
| `OLLAMA_TIMEOUT` | `120.0` | No | Request timeout in seconds |
| `OLLAMA_PULL_MODELS` | `false` | No | Auto-pull missing models |

**Example Configuration:**
```bash
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_LLM_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_DIMENSIONS=1024
OLLAMA_MAX_BATCH_SIZE=5
OLLAMA_TIMEOUT=180.0
OLLAMA_PULL_MODELS=true
```

#### vLLM Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `VLLM_BASE_URL` | - | Yes | vLLM server endpoint URL (must include `/v1`) |
| `VLLM_API_KEY` | `EMPTY` | No | vLLM API key (use "EMPTY" for no auth) |
| `VLLM_TEXT_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | No | Text generation model |
| `VLLM_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | No | Embedding model name |
| `VLLM_EMBEDDING_DIMENSIONS` | `1024` | No | Embedding vector dimensions |
| `VLLM_VISION_MODEL` | `llava-hf/llava-v1.6-mistral-7b-hf` | No | Vision model for image understanding |
| `VLLM_VISION_ENABLED` | `false` | No | Enable vision model capabilities |
| `VLLM_MAX_BATCH_SIZE` | `32` | No | Max texts per batch request |
| `VLLM_MAX_RETRIES` | `3` | No | Retry attempts for failed requests |
| `VLLM_TIMEOUT` | `120.0` | No | Request timeout in seconds |

**Example Configuration:**
```bash
# vLLM Endpoint
VLLM_BASE_URL=https://your-vllm-endpoint.com/v1
VLLM_API_KEY=your_vllm_api_key

# Text and Embedding Models
VLLM_TEXT_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_EMBEDDING_DIMENSIONS=1024

# Vision Model Configuration (optional)
VLLM_VISION_MODEL=llava-hf/llava-v1.6-mistral-7b-hf
VLLM_VISION_ENABLED=true

# Performance Tuning
VLLM_MAX_BATCH_SIZE=32
VLLM_MAX_RETRIES=3
VLLM_TIMEOUT=120.0
```

**Supported Models:**

Text Models:
- `meta-llama/Llama-3.1-8B-Instruct` (default, 8K context)
- `meta-llama/Llama-3.1-70B-Instruct` (high quality, 8K context)
- `mistralai/Mistral-7B-Instruct-v0.3` (fast inference)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (MoE architecture)

Embedding Models:
- `BAAI/bge-large-en-v1.5` (1024 dims, default, best quality)
- `BAAI/bge-base-en-v1.5` (768 dims, balanced)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast)

Vision Models:
- `llava-hf/llava-v1.6-mistral-7b-hf` (default, high quality captions)
- `llava-hf/llava-v1.6-vicuna-7b-hf` (alternative base)
- `Qwen/Qwen2-VL-7B-Instruct` (multilingual support)
- `microsoft/Phi-3-vision-128k-instruct` (long context)

### Vector Database Configuration

#### Database Selection

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `VECTOR_DB_PROVIDER` | `supabase`, `sqlite`, `neo4j_vector`, `pinecone`, `weaviate` | `supabase` | Vector database provider |
| `EMBEDDING_DIMENSION` | Integer | `1536` | Vector embedding dimensions (must match AI provider) |

#### Supabase Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |

**Example:**
```bash
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
EMBEDDING_DIMENSION=1536
```

#### SQLite Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SQLITE_DB_PATH` | `/app/data/vector_db.sqlite` | SQLite database file path |

**Example:**
```bash
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/data/vector_db.sqlite
EMBEDDING_DIMENSION=768
```

#### Neo4j Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Yes | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Yes | Neo4j username |
| `NEO4J_PASSWORD` | - | Yes | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | No | Neo4j database name |

**Example:**
```bash
VECTOR_DB_PROVIDER=neo4j_vector
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=vector_db
```

#### Pinecone Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_ENVIRONMENT` | Yes | Pinecone environment |
| `PINECONE_INDEX_NAME` | Yes | Pinecone index name |

**Example:**
```bash
VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=crawl4ai-rag
```

#### Weaviate Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate server URL |

**Example:**
```bash
VECTOR_DB_PROVIDER=weaviate
WEAVIATE_URL=http://weaviate:8080
```

### RAG Strategy Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_CONTEXTUAL_EMBEDDINGS` | `false` | Enable contextual embedding enhancement |
| `USE_HYBRID_SEARCH` | `false` | Combine vector and keyword search |
| `USE_AGENTIC_RAG` | `false` | Extract and index code examples separately |
| `USE_RERANKING` | `false` | Use cross-encoder reranking |
| `USE_KNOWLEDGE_GRAPH` | `false` | Enable Neo4j knowledge graph features |

**Example Configuration:**
```bash
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=true
```

### Knowledge Graph Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `KG_NEO4J_URI` | `bolt://localhost:7687` | Yes | Neo4j URI for knowledge graph |
| `KG_NEO4J_USER` | `neo4j` | Yes | Neo4j username |
| `KG_NEO4J_PASSWORD` | - | Yes | Neo4j password |
| `KG_NEO4J_DATABASE` | `neo4j` | No | Neo4j database name |

**Example:**
```bash
USE_KNOWLEDGE_GRAPH=true
KG_NEO4J_URI=bolt://neo4j:7687
KG_NEO4J_USER=neo4j
KG_NEO4J_PASSWORD=knowledge_graph_password
KG_NEO4J_DATABASE=knowledge_graph
```

### Legacy Compatibility

| Variable | Maps To | Description |
|----------|---------|-------------|
| `MODEL_CHOICE` | `OPENAI_LLM_MODEL` or `OLLAMA_LLM_MODEL` | Backward compatibility for model selection |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8051` | MCP server port |
| `HOST` | `0.0.0.0` | MCP server host |
| `TRANSPORT_MODE` | `sse` | Transport mode (`sse` or `stdio`) |

## Configuration Examples

### Example 1: Development Setup (Ollama + SQLite)

```bash
# AI Provider
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSIONS=768

# Vector Database
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/app/data/vector_db.sqlite
EMBEDDING_DIMENSION=768

# RAG Strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=true
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Server
PORT=8051
TRANSPORT_MODE=sse
```

### Example 2: Production Setup (OpenAI + Supabase)

```bash
# AI Provider
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-production-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_MAX_BATCH_SIZE=100
OPENAI_TIMEOUT=30.0

# Vector Database
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
EMBEDDING_DIMENSION=1536

# RAG Strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false

# Server
PORT=8051
TRANSPORT_MODE=sse
```

### Example 3: vLLM Cloud Deployment (Text + Vision)

```bash
# AI Provider
AI_PROVIDER=vllm
VLLM_BASE_URL=https://your-vllm-endpoint.com/v1
VLLM_API_KEY=your_vllm_api_key

# Text and Embedding Models
VLLM_TEXT_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_EMBEDDING_DIMENSIONS=1024

# Vision Model (optional)
VLLM_VISION_ENABLED=true
VLLM_VISION_MODEL=llava-hf/llava-v1.6-mistral-7b-hf

# Vector Database
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
EMBEDDING_DIMENSION=1024

# RAG Strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false
```

### Example 4: Cost-Optimized Hybrid (vLLM Embeddings + OpenAI LLM)

```bash
# AI Provider (Mixed)
AI_PROVIDER=mixed
EMBEDDING_PROVIDER=vllm
LLM_PROVIDER=openai

# vLLM for embeddings
VLLM_BASE_URL=https://your-vllm-endpoint.com/v1
VLLM_API_KEY=your_vllm_api_key
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_EMBEDDING_DIMENSIONS=1024

# OpenAI for LLM
OPENAI_API_KEY=sk-your-key
OPENAI_LLM_MODEL=gpt-4o-mini

# Vector Database
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
EMBEDDING_DIMENSION=1024

# RAG Strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

### Example 5: Full-Featured Setup (Neo4j + Knowledge Graph)

```bash
# AI Provider
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536

# Vector Database + Knowledge Graph
VECTOR_DB_PROVIDER=neo4j_vector
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Knowledge Graph
USE_KNOWLEDGE_GRAPH=true
KG_NEO4J_URI=bolt://neo4j:7687
KG_NEO4J_USER=neo4j
KG_NEO4J_PASSWORD=your_password
KG_NEO4J_DATABASE=neo4j

# RAG Strategies
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true

# Embedding Configuration
EMBEDDING_DIMENSION=1536
```

## Environment File Management

### Creating Environment Files

Create separate environment files for different deployments:

```bash
# Development environment
cat > .env.development << EOF
AI_PROVIDER=ollama
VECTOR_DB_PROVIDER=sqlite
OLLAMA_BASE_URL=http://localhost:11434
SQLITE_DB_PATH=/app/data/dev_vector_db.sqlite
USE_CONTEXTUAL_EMBEDDINGS=true
EOF

# Production environment
cat > .env.production << EOF
AI_PROVIDER=openai
VECTOR_DB_PROVIDER=supabase
OPENAI_API_KEY=sk-your-production-key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-production-key
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_RERANKING=true
EOF
```

### Loading Environment Files

```bash
# Load specific environment
source .env.production

# Or use with Docker
docker run --env-file .env.production mcp/crawl4ai-rag

# With docker-compose
docker compose --env-file .env.production up
```

## Configuration Validation

### Built-in Validation

```python
from utils_refactored import validate_ai_provider_config

# Check configuration
result = validate_ai_provider_config()
print("Valid:", result["valid"])
print("Errors:", result.get("errors", []))
print("Warnings:", result.get("warnings", []))
```

### Health Check Script

```bash
# Validate configuration and connectivity
./scripts/health_check.py --provider openai --env production

# Output configuration validation
./scripts/health_check.py --provider ollama --env development --output health.json
```

### Manual Validation Commands

```bash
# Test AI provider connectivity
python -c "
from utils_refactored import create_embedding
try:
    embedding = create_embedding('test')
    print(f'AI Provider: OK ({len(embedding)} dimensions)')
except Exception as e:
    print(f'AI Provider: FAILED - {e}')
"

# Test database connectivity
python -c "
from src.database_factory import get_database_provider
try:
    db = get_database_provider()
    print(f'Database: OK ({type(db).__name__})')
except Exception as e:
    print(f'Database: FAILED - {e}')
"
```

## Configuration Best Practices

### Security

1. **Never commit API keys to version control**
   ```bash
   # Use environment files
   echo ".env*" >> .gitignore
   ```

2. **Use different keys for different environments**
   ```bash
   # Development key (limited quota)
   OPENAI_API_KEY=sk-dev-key...
   
   # Production key (full quota)
   OPENAI_API_KEY=sk-prod-key...
   ```

3. **Rotate API keys regularly**
   ```bash
   # Update keys and restart services
   export OPENAI_API_KEY=sk-new-key...
   docker compose restart
   ```

### Performance

1. **Match embedding dimensions across providers**
   ```bash
   # Ollama setup
   OLLAMA_EMBEDDING_DIMENSIONS=768
   EMBEDDING_DIMENSION=768
   
   # OpenAI setup
   OPENAI_EMBEDDING_DIMENSIONS=1536
   EMBEDDING_DIMENSION=1536
   ```

2. **Adjust batch sizes for provider capabilities**
   ```bash
   # OpenAI can handle larger batches
   OPENAI_MAX_BATCH_SIZE=100
   
   # Ollama works better with smaller batches
   OLLAMA_MAX_BATCH_SIZE=10
   ```

3. **Set appropriate timeouts**
   ```bash
   # OpenAI (faster)
   OPENAI_TIMEOUT=30.0
   
   # Ollama (slower, especially for LLM)
   OLLAMA_TIMEOUT=120.0
   ```

### Monitoring

1. **Enable logging for troubleshooting**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:."
   export LOGLEVEL=DEBUG
   ```

2. **Use health checks in production**
   ```bash
   # Add to cron for monitoring
   */5 * * * * /path/to/scripts/health_check.py --provider openai --env production
   ```

3. **Monitor resource usage**
   ```bash
   # Docker stats
   docker stats
   
   # System resources
   htop
   ```

## Troubleshooting Configuration Issues

### Common Configuration Errors

1. **Missing Required Variables**
   ```bash
   Error: Missing required configuration for openai: ['api_key']
   
   # Solution: Set the API key
   export OPENAI_API_KEY=your-key
   ```

2. **Dimension Mismatch**
   ```bash
   Error: Embedding dimension mismatch: expected 1536, got 768
   
   # Solution: Match dimensions
   export EMBEDDING_DIMENSION=768
   export OLLAMA_EMBEDDING_DIMENSIONS=768
   ```

3. **Provider Not Available**
   ```bash
   Error: Unsupported AI provider: invalid_provider

   # Solution: Use valid provider
   export AI_PROVIDER=openai  # or ollama, vllm, mixed
   ```

4. **Connection Timeout**
   ```bash
   Error: Provider connection failed: timeout
   
   # Solution: Increase timeout or check connectivity
   export OLLAMA_TIMEOUT=180.0
   ```

### Debugging Configuration

1. **Print current configuration**
   ```bash
   env | grep -E "(AI_PROVIDER|OPENAI|OLLAMA|SUPABASE|NEO4J)" | sort
   ```

2. **Test individual components**
   ```bash
   # Test AI provider
   python -c "from utils_refactored import get_ai_provider_info; print(get_ai_provider_info())"
   
   # Test database
   python -c "from src.database_factory import get_database_provider; print(type(get_database_provider()))"
   ```

3. **Use validation tools**
   ```bash
   # Comprehensive validation
   ./scripts/health_check.py --provider your_provider --env your_env --verbose
   ```

This configuration guide provides comprehensive documentation for all supported configuration options and deployment scenarios. Use the examples and validation tools to ensure your setup is correct and optimized for your use case.