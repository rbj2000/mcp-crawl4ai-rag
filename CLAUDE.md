# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that integrates Crawl4AI with multiple vector databases and AI providers for advanced web crawling and RAG (Retrieval Augmented Generation) capabilities. The server enables AI agents and coding assistants to crawl websites, store content in vector databases, and perform intelligent document retrieval with optional AI hallucination detection using Neo4j knowledge graphs.

**Key Features:**
- **Multi-AI Provider Support**: OpenAI, Ollama, and hybrid configurations
- **Multiple Vector Databases**: Supabase, SQLite, Neo4j, Pinecone, Weaviate
- **Flexible Deployment**: Docker orchestration with various provider combinations
- **Cost Optimization**: Mix providers (e.g., OpenAI embeddings + Ollama LLM)
- **Privacy Options**: Fully local deployment with Ollama
- **Enterprise Ready**: Production deployment with monitoring and health checks

## Development Commands

### Docker (Recommended)

**Quick Start with Provider Selection:**
```bash
# OpenAI + Supabase (Production)
docker compose --profile supabase up -d

# Ollama + SQLite (Local Development)
docker compose --profile ollama-sqlite up -d

# Hybrid OpenAI/Ollama + Supabase (Cost-Optimized)
docker compose --profile hybrid up -d

# Full Ollama Stack with Neo4j and Monitoring
docker compose --profile ollama-full up -d
```

**Custom Build:**
```bash
# Build with specific providers
docker build -t mcp/crawl4ai-rag \
  --build-arg AI_PROVIDER=ollama \
  --build-arg VECTOR_DB_PROVIDER=sqlite .

# Run with environment file
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Direct Python Development
```bash
# Install dependencies
uv pip install -e .
crawl4ai-setup

# Configure AI provider (examples)
export AI_PROVIDER=openai  # or ollama, mixed
export OPENAI_API_KEY=your_key
# OR for Ollama
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434

# Run the MCP server
uv run src/crawl4ai_mcp.py
```

### Ollama Setup (for local AI)
```bash
# Install and setup Ollama
./scripts/setup_ollama.py --install --pull-models --optimize balanced

# Start Ollama service
ollama serve
```

### Database Setup
Execute the SQL schema in `crawled_pages.sql` in your Supabase dashboard to create required tables with pgvector extension.

## Architecture

### Core Components

**Main Server** (`src/crawl4ai_mcp.py`):
- FastMCP server with configurable RAG strategies
- Async web crawler using Crawl4AI with browser automation
- Context management for multiple AI providers and vector databases
- Support for multiple transport modes (SSE/stdio)

**AI Provider System** (`src/ai_providers/`):
- Abstract provider interface for consistent API
- OpenAI provider for production-grade embeddings and LLM
- Ollama provider for local, privacy-focused AI processing
- Factory pattern for dynamic provider selection
- Health monitoring and automatic failover

**Utilities** (`src/utils_refactored.py`):
- Multi-provider client management with backward compatibility
- Embedding generation with automatic provider selection
- Smart content chunking respecting code blocks and paragraphs
- Code example extraction and summarization
- Contextual embedding enhancement for better retrieval

**Knowledge Graph System** (`knowledge_graphs/`):
- `parse_repo_into_neo4j.py`: GitHub repository analysis and Neo4j storage
- `ai_script_analyzer.py`: Python AST analysis for code structure
- `knowledge_graph_validator.py`: AI hallucination detection against real repositories
- `hallucination_reporter.py`: Comprehensive validation reporting

### AI Provider Configuration

The server supports multiple AI providers with flexible configuration:

**Provider Options:**
- **OpenAI**: Production-grade, high-accuracy embeddings and LLM
- **Ollama**: Local deployment, privacy-focused, customizable models
- **Hybrid**: Mix providers (e.g., OpenAI embeddings + Ollama LLM for cost optimization)

**Key Benefits:**
- **Cost Control**: Use expensive providers selectively
- **Privacy**: Keep sensitive data local with Ollama
- **Reliability**: Automatic failover between providers
- **Performance**: Choose optimal provider for each task

### RAG Strategy Configuration

The server supports five configurable RAG strategies via environment variables:

1. **USE_CONTEXTUAL_EMBEDDINGS**: Enriches chunks with document context using LLM
2. **USE_HYBRID_SEARCH**: Combines vector similarity with keyword search
3. **USE_AGENTIC_RAG**: Extracts and indexes code examples separately
4. **USE_RERANKING**: Cross-encoder reranking for improved relevance
5. **USE_KNOWLEDGE_GRAPH**: Neo4j-based hallucination detection

### Database Schema

**Supabase Tables**:
- `crawled_pages`: Main content storage with vector embeddings
- `code_examples`: Specialized code block storage with summaries
- `sources`: Source metadata and statistics

**Neo4j Knowledge Graph**:
- Repository → File → Class/Function relationships
- Method and attribute nodes with parameter information
- Import tracking for hallucination detection

### MCP Tools

**Core Tools**:
- `crawl_single_page`: Single URL crawling and storage
- `smart_crawl_url`: Intelligent crawling (sitemap/recursive/text file)
- `get_available_sources`: Source discovery for filtering
- `perform_rag_query`: Semantic search with optional source filtering

**Conditional Tools**:
- `search_code_examples`: Code-specific search (requires USE_AGENTIC_RAG)
- `parse_github_repository`: Repository analysis (requires USE_KNOWLEDGE_GRAPH)
- `check_ai_script_hallucinations`: AI validation (requires USE_KNOWLEDGE_GRAPH)
- `query_knowledge_graph`: Neo4j exploration (requires USE_KNOWLEDGE_GRAPH)

## Configuration

### AI Provider Configuration

**Primary Provider Selection:**
```bash
# Choose primary AI provider
AI_PROVIDER=openai|ollama|mixed

# Or specify providers separately
EMBEDDING_PROVIDER=openai|ollama
LLM_PROVIDER=openai|ollama
```

**OpenAI Configuration:**
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536
```

**Ollama Configuration:**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSIONS=768
```

### Vector Database Configuration

**Database Selection:**
```bash
VECTOR_DB_PROVIDER=supabase|sqlite|neo4j_vector|pinecone|weaviate
```

**Supabase:**
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
```

**SQLite (Local Development):**
```bash
SQLITE_DB_PATH=/app/data/vector_db.sqlite
```

**Neo4j (Knowledge Graph + Vector):**
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### Legacy Configuration

For backward compatibility:
```bash
MODEL_CHOICE=gpt-4o-mini  # Maps to LLM_PROVIDER selection
```

## Development Notes

- **Multi-Provider Support**: Seamlessly switch between OpenAI, Ollama, or hybrid configurations
- **Backward Compatibility**: Existing `utils.py` usage works unchanged with `utils_refactored.py`
- **Performance**: Batch processing and connection pooling optimized for each provider
- **Content Chunking**: Respects markdown structure and code blocks across all providers
- **Retry Logic**: Exponential backoff with provider-specific error handling
- **Health Monitoring**: Automatic provider health checks and failover
- **Docker Orchestration**: Full multi-provider deployment support via docker-compose profiles
- **Testing**: Comprehensive test suite covering all provider combinations

## Quick Start Examples

### Local Development (Ollama)
```bash
# Setup and start
./scripts/setup_ollama.py --install --pull-models
docker compose --profile ollama-sqlite up -d

# Test endpoint
curl http://localhost:8056/health
```

### Production (OpenAI + Supabase)
```bash
# Configure environment
export OPENAI_API_KEY=your_key
export SUPABASE_URL=your_url
export SUPABASE_SERVICE_KEY=your_key

# Deploy
docker compose --profile supabase up -d
```

### Cost-Optimized (Hybrid)
```bash
# OpenAI embeddings + Ollama LLM
export OPENAI_API_KEY=your_key
docker compose --profile hybrid up -d
```

## Migration from OpenAI-Only

1. **Update imports** (optional, for direct usage):
   ```python
   # Old
   from utils import create_embedding
   # New  
   from utils_refactored import create_embedding
   ```

2. **Set provider environment variables**:
   ```bash
   export AI_PROVIDER=openai  # or ollama, mixed
   ```

3. **No other code changes required** - full backward compatibility maintained

## Health Checks and Monitoring

```bash
# Check deployment health
./scripts/health_check.py --provider ollama --env development

# Monitor metrics
curl http://localhost:8051/metrics

# Provider-specific validation
./scripts/setup_ollama.py --validate
```