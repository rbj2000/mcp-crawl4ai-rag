# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Model Context Protocol (MCP) server that integrates Crawl4AI with Supabase for advanced web crawling and RAG (Retrieval Augmented Generation) capabilities. The server enables AI agents and coding assistants to crawl websites, store content in vector databases, and perform intelligent document retrieval with optional AI hallucination detection using Neo4j knowledge graphs.

## Development Commands

### Docker (Recommended)
```bash
# Build the Docker image
docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .

# Run with environment file
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Direct Python Development
```bash
# Install dependencies
uv pip install -e .
crawl4ai-setup

# Run the MCP server
uv run src/crawl4ai_mcp.py
```

### Database Setup
Execute the SQL schema in `crawled_pages.sql` in your Supabase dashboard to create required tables with pgvector extension.

## Architecture

### Core Components

**Main Server** (`src/crawl4ai_mcp.py`):
- FastMCP server with configurable RAG strategies
- Async web crawler using Crawl4AI with browser automation
- Context management for crawler, Supabase client, and Neo4j components
- Support for multiple transport modes (SSE/stdio)

**Utilities** (`src/utils.py`):
- Supabase client management and vector operations
- OpenAI embedding generation with batch processing and retry logic
- Smart content chunking respecting code blocks and paragraphs
- Code example extraction and summarization
- Contextual embedding enhancement for better retrieval

**Knowledge Graph System** (`knowledge_graphs/`):
- `parse_repo_into_neo4j.py`: GitHub repository analysis and Neo4j storage
- `ai_script_analyzer.py`: Python AST analysis for code structure
- `knowledge_graph_validator.py`: AI hallucination detection against real repositories
- `hallucination_reporter.py`: Comprehensive validation reporting

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

Required environment variables in `.env`:
- `OPENAI_API_KEY`: For embeddings and LLM processing
- `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`: Vector database
- `MODEL_CHOICE`: LLM model for summaries and contextual embeddings
- Optional Neo4j credentials for knowledge graph features

## Development Notes

- The system uses OpenAI's `text-embedding-3-small` for all vector embeddings
- Content chunking respects markdown structure and code blocks
- Parallel processing is used extensively for performance (embeddings, crawling, analysis)
- Retry logic with exponential backoff for external API calls
- Knowledge graph functionality requires Neo4j but can be disabled
- Docker deployment doesn't fully support knowledge graph features yet