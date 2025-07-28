# Local Testing Guide - MCP Crawl4AI RAG

This guide shows you how to test the MCP server locally without Docker, using SQLite and Neo4j.

## Prerequisites ✅

- [x] Virtual environment created and activated
- [x] All dependencies installed with `uv pip install -e ".[all]"`
- [x] Crawl4AI setup completed with `crawl4ai-setup`
- [x] Neo4j container running (optional, for knowledge graph features)

## Configuration Options

You have two main options for AI providers:

### Option 1: OpenAI (Recommended for Testing)

1. **Get an OpenAI API key** from https://platform.openai.com/api-keys

2. **Update `.env.local`**:
   ```bash
   # Replace 'your_openai_api_key_here' with your actual API key
   sed -i '' 's/your_openai_api_key_here/YOUR_ACTUAL_API_KEY/' .env.local
   ```

### Option 2: Local Ollama (Privacy-Focused)

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

2. **Start Ollama and pull models**:
   ```bash
   ollama serve &
   ollama pull nomic-embed-text    # For embeddings
   ollama pull llama3.2           # For LLM tasks
   ```

3. **Update `.env.local`** to use Ollama:
   ```bash
   # Comment out OpenAI settings and uncomment Ollama settings
   AI_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   OLLAMA_LLM_MODEL=llama3.2
   OLLAMA_EMBEDDING_DIMENSIONS=768
   ```

## Quick Start

### 1. Activate Environment and Test Configuration

```bash
# Activate virtual environment
source .venv/bin/activate

# Test your configuration
python test_local.py
```

You should see:
```
✅ All tests passed! Your local setup is ready.
```

### 2. Start the MCP Server

```bash
# Load environment variables and start server
source .venv/bin/activate
export $(cat .env.local | grep -v '^#' | grep '=' | xargs)
python src/crawl4ai_mcp.py
```

The server should start and show:
```
MCP Server running on http://0.0.0.0:8051
Available at: http://localhost:8051/sse
```

### 3. Connect with Claude Code

```bash
# Add the server to Claude Code
claude mcp add-json crawl4ai-rag-local '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

## Testing the Server

### Basic Web Crawling Test

Once connected to Claude Code, try these commands:

1. **Test single page crawling**:
   ```
   Please crawl this page: https://docs.python.org/3/library/os.html
   ```

2. **Test smart crawling**:
   ```
   Please crawl the FastAPI documentation: https://fastapi.tiangolo.com/
   ```

3. **Test RAG search**:
   ```
   Search for information about "file operations" in the crawled content
   ```

4. **Check available sources**:
   ```
   What sources are available in the database?
   ```

### Knowledge Graph Testing (if enabled)

If you have `USE_KNOWLEDGE_GRAPH=true` in your `.env.local`:

1. **Add a repository to the knowledge graph**:
   ```
   Please add https://github.com/pydantic/pydantic-ai.git to the knowledge graph
   ```

2. **Query the knowledge graph**:
   ```
   What classes are available in the pydantic-ai repository?
   ```

3. **Test hallucination detection**:
   ```
   Please check this Python code for hallucinations:
   
   from pydantic_ai import Agent
   agent = Agent()
   result = agent.run_fake_method("test")
   ```

## Monitoring

### Neo4j Browser (if using knowledge graph)
- Open http://localhost:7474 in your browser
- Login with: `neo4j` / `testpassword`
- Run queries like: `MATCH (n) RETURN n LIMIT 25`

### SQLite Database
Your crawled data is stored in `./local_test.db`. You can inspect it with:
```bash
# Install sqlite3 if needed
brew install sqlite3

# View tables
sqlite3 local_test.db ".tables"

# View crawled pages
sqlite3 local_test.db "SELECT url, title FROM crawled_pages LIMIT 5;"
```

## Troubleshooting

### Common Issues

1. **"OpenAI API Key not set"**
   - Update `.env.local` with your actual API key
   - Or switch to Ollama configuration

2. **"Neo4j connection failed"**
   - Check if Neo4j container is running: `docker ps | grep neo4j`
   - Restart if needed: `docker restart neo4j-test`

3. **"Import errors"**
   - Make sure virtual environment is activated: `source .venv/bin/activate`
   - Reinstall dependencies: `uv pip install -e ".[all]"`

4. **"Server won't start"**
   - Check if port 8051 is available: `lsof -i :8051`
   - Kill existing processes: `pkill -f crawl4ai_mcp`

### Debug Mode

For more detailed logging, set:
```bash
export LOG_LEVEL=DEBUG
python src/crawl4ai_mcp.py
```

## Configuration Reference

### Database Providers
- `sqlite` - Local file database (default for testing)
- `supabase` - Cloud PostgreSQL with pgvector
- `pinecone` - Managed vector database
- `neo4j_vector` - Neo4j with vector search
- `weaviate` - Enterprise vector database

### RAG Strategies
- `USE_CONTEXTUAL_EMBEDDINGS` - Enhanced chunk context (slower, more accurate)
- `USE_HYBRID_SEARCH` - Vector + keyword search (recommended)
- `USE_AGENTIC_RAG` - Separate code example indexing (for coding docs)
- `USE_RERANKING` - Improved result ordering (recommended)
- `USE_KNOWLEDGE_GRAPH` - AI hallucination detection (requires Neo4j)

### Performance Tips

1. **For fast testing**: Disable `USE_CONTEXTUAL_EMBEDDINGS` and `USE_AGENTIC_RAG`
2. **For best results**: Enable `USE_HYBRID_SEARCH` and `USE_RERANKING`
3. **For coding docs**: Enable `USE_AGENTIC_RAG` to get code-specific search tool
4. **For AI validation**: Enable `USE_KNOWLEDGE_GRAPH` for hallucination detection

## Next Steps

Once you have the server running locally:

1. **Test different websites** - Try crawling various documentation sites
2. **Experiment with RAG strategies** - Enable/disable different features in `.env.local`
3. **Test knowledge graph features** - Add repositories and test hallucination detection
4. **Performance tuning** - Adjust batch sizes and timeouts based on your needs

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python test_local.py` to validate your setup
3. Check the server logs for detailed error messages
4. Ensure all prerequisites are properly installed