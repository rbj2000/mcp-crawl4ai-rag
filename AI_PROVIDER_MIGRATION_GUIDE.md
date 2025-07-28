# AI Provider Migration Guide

This guide helps users migrate from the OpenAI-only version of the MCP Crawl4AI RAG system to the new multi-provider architecture supporting OpenAI, Ollama, and hybrid configurations.

## Overview

The multi-provider update introduces:
- **Flexible AI Provider Selection**: Choose between OpenAI, Ollama, or hybrid configurations
- **Cost Optimization**: Mix providers to balance cost and performance
- **Privacy Options**: Fully local deployment with Ollama
- **Backward Compatibility**: Existing configurations continue to work unchanged

## Migration Strategies

### Strategy 1: Zero-Downtime Migration (Recommended)

**For users who want to keep using OpenAI with no changes:**

1. **No action required** - your existing configuration continues to work
2. Optionally set explicit provider variables for clarity:
   ```bash
   export AI_PROVIDER=openai
   ```

**Benefits**: 
- No downtime
- No configuration changes needed
- Identical behavior to current setup

### Strategy 2: Gradual Migration to Ollama

**For users who want to migrate to local AI over time:**

**Phase 1: Setup Ollama alongside existing OpenAI**
```bash
# Install Ollama
./scripts/setup_ollama.py --install --pull-models

# Test Ollama independently
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
./scripts/health_check.py --provider ollama --env development
```

**Phase 2: Switch to hybrid mode (OpenAI embeddings + Ollama LLM)**
```bash
# Cost-optimized hybrid configuration
export AI_PROVIDER=mixed
export EMBEDDING_PROVIDER=openai  
export LLM_PROVIDER=ollama
export OPENAI_API_KEY=your_existing_key
export OLLAMA_BASE_URL=http://localhost:11434
```

**Phase 3: Full Ollama migration**
```bash
# Complete local setup
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_LLM_MODEL=llama3.2:1b
```

### Strategy 3: Direct Migration to Hybrid

**For users who want immediate cost optimization:**

```bash
# Hybrid setup from the start
export AI_PROVIDER=mixed
export EMBEDDING_PROVIDER=openai
export LLM_PROVIDER=ollama
export OPENAI_API_KEY=your_existing_key

# Install Ollama for LLM usage
./scripts/setup_ollama.py --install --pull-models
```

## Step-by-Step Migration

### Step 1: Backup Current Configuration

```bash
# Backup your current environment
cp .env .env.backup.$(date +%Y%m%d)

# Export current settings for reference
env | grep -E "(OPENAI|SUPABASE|MODEL)" > current_config.txt
```

### Step 2: Choose Your Target Configuration

#### Option A: Stay with OpenAI (No Changes)
```bash
# Optional: Make provider explicit
echo "AI_PROVIDER=openai" >> .env
```

#### Option B: Migrate to Ollama
```bash
# Install Ollama
./scripts/setup_ollama.py --install --pull-models --optimize balanced

# Update configuration
cat >> .env << EOF
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSIONS=768
EMBEDDING_DIMENSION=768
EOF
```

#### Option C: Hybrid Configuration
```bash
# Install Ollama for LLM
./scripts/setup_ollama.py --install --pull-models

# Configure hybrid mode
cat >> .env << EOF
AI_PROVIDER=mixed
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:1b
EOF
```

### Step 3: Update Code (Optional)

**For direct Python usage**, update imports:
```python
# Old import
from utils import create_embedding, create_embeddings_batch

# New import (backward compatible)
from utils_refactored import create_embedding, create_embeddings_batch
```

**Note**: The MCP server automatically uses the new provider system - no code changes needed for MCP usage.

### Step 4: Validate Configuration

```bash
# Test configuration
python -c "
from utils_refactored import validate_ai_provider_config
result = validate_ai_provider_config()
print('Configuration valid:', result['valid'])
print('Provider:', result.get('provider', 'Not specified'))
"

# Test basic functionality
python -c "
from utils_refactored import create_embedding
try:
    embedding = create_embedding('test')
    print(f'Success: {len(embedding)} dimensions')
except Exception as e:
    print(f'Error: {e}')
"
```

### Step 5: Deploy and Test

#### Docker Deployment
```bash
# OpenAI + Supabase
docker compose --profile supabase up -d

# Ollama + SQLite
docker compose --profile ollama-sqlite up -d

# Hybrid + Supabase
docker compose --profile hybrid up -d
```

#### Direct Deployment
```bash
# Start the server
uv run src/crawl4ai_mcp.py

# Test health endpoint
curl http://localhost:8051/health
```

### Step 6: Verify Functionality

```bash
# Comprehensive health check
./scripts/health_check.py --provider your_provider --env your_env

# Test key MCP operations
# (Use your preferred MCP client to test)
# - crawl_single_page
# - perform_rag_query
# - get_available_sources
```

## Configuration Migration Examples

### Example 1: OpenAI-Only to OpenAI-Only (Explicit)

**Before:**
```bash
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJ...
MODEL_CHOICE=gpt-4o-mini
```

**After:**
```bash
# Same configuration, optionally add:
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJ...
MODEL_CHOICE=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
```

### Example 2: OpenAI to Ollama Migration

**Before:**
```bash
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJ...
MODEL_CHOICE=gpt-4o-mini
```

**After:**
```bash
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSIONS=768
EMBEDDING_DIMENSION=768
# Keep Supabase or switch to SQLite
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/app/data/vector_db.sqlite
```

### Example 3: Hybrid Configuration

**Before:**
```bash
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJ...
MODEL_CHOICE=gpt-4o-mini
```

**After:**
```bash
AI_PROVIDER=mixed
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=ollama
OPENAI_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2:1b
SUPABASE_URL=https://...
SUPABASE_SERVICE_KEY=eyJ...
EMBEDDING_DIMENSION=1536  # OpenAI embedding dimension
```

## Docker Migration

### Update Docker Compose Usage

**Before:**
```bash
docker build -t mcp/crawl4ai-rag .
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

**After - Profile-Based Deployment:**
```bash
# OpenAI + Supabase (same as before)
docker compose --profile supabase up -d

# Ollama + SQLite (new option)
docker compose --profile ollama-sqlite up -d

# Hybrid configuration (new option)
docker compose --profile hybrid up -d
```

### Custom Docker Builds

**Multi-stage builds with provider selection:**
```bash
# OpenAI build
docker build -t mcp/crawl4ai-rag:openai \
  --build-arg AI_PROVIDER=openai \
  --build-arg VECTOR_DB_PROVIDER=supabase .

# Ollama build
docker build -t mcp/crawl4ai-rag:ollama \
  --build-arg AI_PROVIDER=ollama \
  --build-arg VECTOR_DB_PROVIDER=sqlite .

# Hybrid build
docker build -t mcp/crawl4ai-rag:hybrid \
  --build-arg AI_PROVIDER=mixed \
  --build-arg VECTOR_DB_PROVIDER=supabase .
```

## Troubleshooting Migration Issues

### Issue 1: Ollama Connection Failed

**Symptom:**
```
Error: Failed to connect to Ollama at http://localhost:11434
```

**Solutions:**
```bash
# Check if Ollama is running
./scripts/setup_ollama.py --status

# Start Ollama service
ollama serve

# Or use Docker
docker compose --profile ollama up -d
```

### Issue 2: Model Not Found

**Symptom:**
```
Error: Model 'nomic-embed-text' not found
```

**Solutions:**
```bash
# Pull required models
./scripts/setup_ollama.py --pull-models

# Or manually
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

### Issue 3: Dimension Mismatch

**Symptom:**
```
Error: Embedding dimension mismatch: expected 1536, got 768
```

**Solutions:**
```bash
# For Ollama (768 dimensions)
export EMBEDDING_DIMENSION=768
export OLLAMA_EMBEDDING_DIMENSIONS=768

# For OpenAI (1536 dimensions)
export EMBEDDING_DIMENSION=1536
export OPENAI_EMBEDDING_DIMENSIONS=1536
```

### Issue 4: Configuration Validation Failed

**Symptom:**
```
Error: Invalid AI provider configuration
```

**Solutions:**
```bash
# Check configuration
python -c "
from utils_refactored import validate_ai_provider_config
print(validate_ai_provider_config())
"

# Check required environment variables
./scripts/health_check.py --provider your_provider --env your_env
```

### Issue 5: Performance Degradation

**Symptom:**
Slower response times after migration to Ollama

**Solutions:**
```bash
# Apply performance optimization
./scripts/setup_ollama.py --optimize high_performance

# Monitor resource usage
docker stats

# Adjust Ollama settings
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
```

## Rollback Procedures

### Quick Rollback to OpenAI

If you need to quickly rollback to OpenAI-only:

```bash
# Restore backup
cp .env.backup.$(date +%Y%m%d) .env

# Or set explicit OpenAI configuration
export AI_PROVIDER=openai
export OPENAI_API_KEY=your_key

# Restart services
docker compose down
docker compose --profile supabase up -d
```

### Gradual Rollback

For hybrid configurations, you can rollback gradually:

```bash
# Step 1: Switch LLM back to OpenAI (keep Ollama embeddings)
export LLM_PROVIDER=openai

# Step 2: Switch embeddings back to OpenAI
export EMBEDDING_PROVIDER=openai

# Step 3: Remove Ollama entirely
export AI_PROVIDER=openai
```

## Performance Comparison

### Benchmarking Your Migration

Test performance before and after migration:

```bash
# Run performance tests
python -m pytest tests/test_performance_comparison.py -v

# Or use the test runner
python tests/test_runner.py --test-type performance --provider openai
python tests/test_runner.py --test-type performance --provider ollama
```

**Expected Performance Characteristics:**

| Provider | Embedding Speed | LLM Speed | Cost | Privacy |
|----------|----------------|-----------|------|---------|
| OpenAI | Fast | Fast | High | Low |
| Ollama | Medium | Medium | Free | High |
| Hybrid | Fast | Medium | Medium | Medium |

## Post-Migration Checklist

- [ ] Configuration validated successfully
- [ ] Basic embedding generation works
- [ ] MCP server starts without errors
- [ ] Health checks pass
- [ ] Key MCP tools function correctly
- [ ] Performance meets requirements
- [ ] Backup of old configuration created
- [ ] Documentation updated for team
- [ ] Monitoring configured (if applicable)
- [ ] Rollback procedure tested

## Getting Help

If you encounter issues during migration:

1. **Check the troubleshooting section** above for common issues
2. **Run diagnostic commands**:
   ```bash
   ./scripts/health_check.py --provider your_provider --env your_env
   ./scripts/setup_ollama.py --validate  # For Ollama issues
   ```
3. **Review logs**:
   ```bash
   docker compose logs -f your_service_name
   ```
4. **Test with minimal configuration** to isolate issues
5. **Create an issue** with detailed error logs and configuration

## Advanced Migration Scenarios

### Migrating with Custom Models

If you're using custom OpenAI models or want to use specific Ollama models:

```bash
# Custom OpenAI models
export OPENAI_EMBEDDING_MODEL=text-embedding-3-large
export OPENAI_LLM_MODEL=gpt-4-turbo

# Custom Ollama models
export OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
export OLLAMA_LLM_MODEL=llama3.1:8b
```

### Migrating with Multiple Deployments

For teams running multiple environments:

```bash
# Development (Ollama)
docker compose --profile ollama-sqlite up -d

# Staging (Hybrid)
docker compose --profile hybrid up -d

# Production (OpenAI)
docker compose --profile supabase up -d
```

### Migrating with Monitoring

Enable monitoring during migration:

```bash
# Deploy with monitoring
docker compose --profile hybrid --profile monitoring up -d

# Access monitoring dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
```

This completes the comprehensive migration guide. The multi-provider architecture maintains full backward compatibility while providing flexibility for cost optimization, privacy, and performance tuning.