# AI Provider Troubleshooting Guide

This guide provides comprehensive troubleshooting for the MCP Crawl4AI RAG system with multi-provider AI support, covering OpenAI, Ollama, and hybrid configurations.

## Quick Diagnostic Commands

Before diving into specific issues, run these diagnostic commands to gather information:

```bash
# System health check
./scripts/health_check.py --provider your_provider --env your_env

# Configuration validation
python -c "from utils_refactored import validate_ai_provider_config; print(validate_ai_provider_config())"

# Provider information
python -c "from utils_refactored import get_ai_provider_info; print(get_ai_provider_info())"

# Test basic functionality
python -c "from utils_refactored import create_embedding; print(len(create_embedding('test')))"
```

## Common Issues by Category

### Configuration Issues

#### Issue: Missing Required Configuration

**Symptoms:**
```
Error: Missing required configuration for openai: ['api_key']
Error: Invalid AI provider configuration
```

**Diagnosis:**
```bash
# Check environment variables
env | grep -E "(AI_PROVIDER|OPENAI|OLLAMA)" | sort

# Validate configuration
python -c "from utils_refactored import validate_ai_provider_config; print(validate_ai_provider_config())"
```

**Solutions:**
```bash
# For OpenAI
export OPENAI_API_KEY=your_api_key

# For Ollama
export OLLAMA_BASE_URL=http://localhost:11434

# For hybrid
export AI_PROVIDER=mixed
export EMBEDDING_PROVIDER=openai
export LLM_PROVIDER=ollama
export OPENAI_API_KEY=your_key
export OLLAMA_BASE_URL=http://localhost:11434
```

#### Issue: Provider Not Supported

**Symptoms:**
```
Error: Unsupported AI provider: invalid_provider
```

**Solutions:**
```bash
# Check available providers
python -c "from utils_refactored import get_available_ai_providers; print(get_available_ai_providers())"

# Set valid provider
export AI_PROVIDER=openai  # or ollama, mixed
```

#### Issue: Dimension Mismatch

**Symptoms:**
```
Error: Embedding dimension mismatch: expected 1536, got 768
Error: Vector dimension inconsistency
```

**Diagnosis:**
```bash
# Check current dimensions
python -c "
import os
print('EMBEDDING_DIMENSION:', os.getenv('EMBEDDING_DIMENSION', 'Not set'))
print('OPENAI_EMBEDDING_DIMENSIONS:', os.getenv('OPENAI_EMBEDDING_DIMENSIONS', 'Not set'))
print('OLLAMA_EMBEDDING_DIMENSIONS:', os.getenv('OLLAMA_EMBEDDING_DIMENSIONS', 'Not set'))
"

# Test actual embedding dimensions
python -c "
from utils_refactored import create_embedding
embedding = create_embedding('test')
print(f'Actual dimensions: {len(embedding)}')
"
```

**Solutions:**
```bash
# For OpenAI (1536 dimensions)
export EMBEDDING_DIMENSION=1536
export OPENAI_EMBEDDING_DIMENSIONS=1536

# For Ollama (768 dimensions for nomic-embed-text)
export EMBEDDING_DIMENSION=768
export OLLAMA_EMBEDDING_DIMENSIONS=768

# For hybrid (match embedding provider)
export EMBEDDING_DIMENSION=1536  # If using OpenAI embeddings
```

### OpenAI-Specific Issues

#### Issue: OpenAI API Authentication Failed

**Symptoms:**
```
Error: Incorrect API key provided
Error: OpenAI API request failed with status 401
```

**Diagnosis:**
```bash
# Test API key directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check API key format
echo $OPENAI_API_KEY | cut -c1-10
# Should start with 'sk-'
```

**Solutions:**
```bash
# Set correct API key
export OPENAI_API_KEY=sk-your-correct-api-key

# Check for extra whitespace
export OPENAI_API_KEY=$(echo $OPENAI_API_KEY | tr -d ' \n\r')

# Verify key works
python -c "
import openai
openai.api_key = '$OPENAI_API_KEY'
try:
    models = openai.models.list()
    print('API key valid')
except Exception as e:
    print(f'API key invalid: {e}')
"
```

#### Issue: OpenAI Rate Limiting

**Symptoms:**
```
Error: Rate limit exceeded for model
Error: OpenAI API request failed with status 429
```

**Diagnosis:**
```bash
# Check rate limit headers in debug mode
export LOGLEVEL=DEBUG
python -c "from utils_refactored import create_embedding; create_embedding('test')"
```

**Solutions:**
```bash
# Reduce batch size
export OPENAI_MAX_BATCH_SIZE=50

# Increase timeout and retries
export OPENAI_TIMEOUT=60.0
export OPENAI_MAX_RETRIES=5

# Implement delays between requests (in code)
# The utils_refactored already includes exponential backoff
```

#### Issue: OpenAI Model Not Found

**Symptoms:**
```
Error: Model 'text-embedding-3-small' not found
Error: Invalid model specified
```

**Solutions:**
```bash
# Check available models
python -c "
import openai
openai.api_key = '$OPENAI_API_KEY'
models = openai.models.list()
for model in models.data:
    if 'embed' in model.id.lower():
        print(model.id)
"

# Use correct model names
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small  # or text-embedding-ada-002
export OPENAI_LLM_MODEL=gpt-4o-mini  # or gpt-3.5-turbo
```

### Ollama-Specific Issues

#### Issue: Ollama Connection Failed

**Symptoms:**
```
Error: Failed to connect to Ollama at http://localhost:11434
Error: Connection refused
```

**Diagnosis:**
```bash
# Check if Ollama is running
./scripts/setup_ollama.py --status

# Test connection directly
curl http://localhost:11434/api/tags

# Check Docker (if using Docker)
docker ps | grep ollama
```

**Solutions:**
```bash
# Start Ollama service
ollama serve

# Or use Docker
docker compose --profile ollama up -d

# Check firewall/port access
netstat -an | grep 11434

# Test with different URL
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

#### Issue: Ollama Model Not Found

**Symptoms:**
```
Error: Model 'nomic-embed-text' not found
Error: model not found, try pulling it first
```

**Diagnosis:**
```bash
# List available models
ollama list

# Check model status
curl http://localhost:11434/api/tags
```

**Solutions:**
```bash
# Pull required models
./scripts/setup_ollama.py --pull-models

# Or manually
ollama pull nomic-embed-text
ollama pull llama3.2:1b

# Enable auto-pull
export OLLAMA_PULL_MODELS=true
```

#### Issue: Ollama Performance Problems

**Symptoms:**
```
Error: Request timeout
Very slow response times
High memory usage
```

**Diagnosis:**
```bash
# Check system resources
docker stats  # If using Docker
htop

# Check Ollama logs
docker logs ollama  # If using Docker
```

**Solutions:**
```bash
# Optimize for your system
./scripts/setup_ollama.py --optimize balanced  # or low_memory, high_performance

# Adjust timeouts
export OLLAMA_TIMEOUT=180.0

# Reduce batch size
export OLLAMA_MAX_BATCH_SIZE=5

# Use smaller models
export OLLAMA_LLM_MODEL=llama3.2:1b  # instead of larger models
```

#### Issue: Ollama Memory Issues

**Symptoms:**
```
Error: out of memory
Container killed (OOMKilled)
System becoming unresponsive
```

**Solutions:**
```bash
# Apply low memory optimization
./scripts/setup_ollama.py --optimize low_memory

# Use smaller models
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text  # (small)
export OLLAMA_LLM_MODEL=llama3.2:1b  # (1B parameters)

# Limit Docker memory
# Add to docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 4G

# Adjust Ollama settings
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

### Hybrid Configuration Issues

#### Issue: Mixed Provider Configuration Errors

**Symptoms:**
```
Error: Inconsistent provider configuration
Error: Both embedding and LLM providers must be specified
```

**Solutions:**
```bash
# Explicit hybrid configuration
export AI_PROVIDER=mixed
export EMBEDDING_PROVIDER=openai
export LLM_PROVIDER=ollama

# Verify configuration
python -c "
from utils_refactored import get_ai_provider_info
print(get_ai_provider_info())
"
```

#### Issue: Hybrid Performance Issues

**Symptoms:**
- Fast embeddings, slow LLM responses
- Inconsistent response times

**Solutions:**
```bash
# Optimize each provider separately
# OpenAI embeddings
export OPENAI_MAX_BATCH_SIZE=100
export OPENAI_TIMEOUT=30.0

# Ollama LLM
export OLLAMA_TIMEOUT=120.0
./scripts/setup_ollama.py --optimize balanced
```

### Database Connection Issues

#### Issue: Vector Database Connection Failed

**Symptoms:**
```
Error: Failed to connect to Supabase
Error: SQLite database not accessible
Error: Neo4j connection timeout
```

**Diagnosis:**
```bash
# Test database connection
python -c "
from src.database_factory import get_database_provider
try:
    db = get_database_provider()
    print(f'Database connection: OK ({type(db).__name__})')
except Exception as e:
    print(f'Database connection: FAILED - {e}')
"
```

**Solutions:**
```bash
# For Supabase
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_SERVICE_KEY=eyJ...

# Test Supabase connection
curl "$SUPABASE_URL/rest/v1/" -H "apikey: $SUPABASE_SERVICE_KEY"

# For SQLite
export SQLITE_DB_PATH=/app/data/vector_db.sqlite
mkdir -p /app/data

# For Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

### Docker-Specific Issues

#### Issue: Docker Container Health Checks Failing

**Symptoms:**
```
Container marked as unhealthy
Health check failing
```

**Diagnosis:**
```bash
# Check container status
docker compose ps

# Check health check logs
docker inspect container_name | grep -A 10 Health

# Manual health check
docker exec container_name curl -f http://localhost:8051/health
```

**Solutions:**
```bash
# Increase health check timeout
# In docker-compose.yml:
# healthcheck:
#   timeout: 30s
#   start_period: 120s

# Check application logs
docker compose logs -f your_service

# Restart unhealthy containers
docker compose restart your_service
```

#### Issue: Docker Build Failures

**Symptoms:**
```
Error building Docker image
Build context size issues
```

**Solutions:**
```bash
# Clean build cache
docker system prune -f

# Use multi-stage build
docker build -f Dockerfile.multi-stage -t mcp/crawl4ai-rag .

# Check .dockerignore
echo "*.pyc
__pycache__
.git
.env*" > .dockerignore
```

## Environment-Specific Troubleshooting

### Development Environment

**Common Issues:**
- Model downloads failing
- Port conflicts
- Resource constraints

**Solutions:**
```bash
# Use lighter models
export OLLAMA_LLM_MODEL=llama3.2:1b
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Alternative ports
export PORT=8052

# Enable debug logging
export LOGLEVEL=DEBUG
```

### Production Environment

**Common Issues:**
- Rate limiting
- Resource exhaustion
- Network connectivity

**Solutions:**
```bash
# Production-optimized configuration
export OPENAI_MAX_BATCH_SIZE=100
export OPENAI_TIMEOUT=30.0
export OPENAI_MAX_RETRIES=5

# Health monitoring
./scripts/health_check.py --provider openai --env production --output health.json

# Log rotation
docker compose logs --since=24h > logs/$(date +%Y%m%d).log
```

## Advanced Troubleshooting

### Network Debugging

```bash
# Test network connectivity
ping api.openai.com
curl -I https://api.openai.com/v1/models

# Check Docker networking
docker network ls
docker network inspect mcp-crawl4ai-rag_default

# Test internal container communication
docker exec container1 ping container2
```

### Performance Profiling

```bash
# Profile embedding generation
python -c "
import time
from utils_refactored import create_embeddings_batch

texts = ['test'] * 100
start = time.time()
embeddings = create_embeddings_batch(texts)
end = time.time()
print(f'Time: {end-start:.2f}s, Rate: {len(texts)/(end-start):.1f} texts/sec')
"

# Monitor resource usage
docker stats --no-stream

# Profile memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

### Log Analysis

```bash
# Enable debug logging
export LOGLEVEL=DEBUG

# Structured log analysis
docker compose logs your_service 2>&1 | grep ERROR
docker compose logs your_service 2>&1 | grep -i "timeout\|connection\|failed"

# Real-time monitoring
docker compose logs -f your_service | grep -E "(ERROR|WARN|Failed)"
```

## Error Code Reference

### HTTP Status Codes

| Code | Provider | Meaning | Solution |
|------|----------|---------|----------|
| 401 | OpenAI | Invalid API key | Check OPENAI_API_KEY |
| 429 | OpenAI | Rate limit exceeded | Reduce batch size, add delays |
| 500 | OpenAI | Server error | Retry with exponential backoff |
| 503 | Ollama | Service unavailable | Check Ollama service status |
| 404 | Ollama | Model not found | Pull model with ollama pull |

### Custom Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| PROVIDER_NOT_CONFIGURED | AI provider not set up | Set AI_PROVIDER environment variable |
| DIMENSION_MISMATCH | Embedding dimensions don't match | Align EMBEDDING_DIMENSION with provider |
| MODEL_NOT_AVAILABLE | Requested model not accessible | Check model name and availability |
| CONNECTION_TIMEOUT | Provider connection timed out | Increase timeout, check connectivity |

## Recovery Procedures

### Automatic Recovery

The system includes automatic recovery mechanisms:
- Exponential backoff for failed requests
- Provider health checks and failover
- Connection pooling and retry logic

### Manual Recovery

```bash
# Reset provider connections
python -c "
from utils_refactored import close_ai_providers, initialize_ai_providers
import asyncio
async def reset():
    await close_ai_providers()
    await initialize_ai_providers()
asyncio.run(reset())
"

# Restart services
docker compose restart

# Clear cache/state
docker compose down -v  # Warning: removes volumes
docker compose up -d
```

### Emergency Rollback

```bash
# Quick rollback to OpenAI
export AI_PROVIDER=openai
export OPENAI_API_KEY=your_backup_key
docker compose restart

# Restore from backup
cp .env.backup .env
source .env
```

## Prevention Strategies

### Monitoring Setup

```bash
# Continuous health monitoring
# Add to cron:
*/5 * * * * /path/to/scripts/health_check.py --provider your_provider --env your_env >/dev/null 2>&1

# Alert on failures
# ./scripts/health_check.py can output JSON for monitoring systems
```

### Best Practices

1. **Configuration Management**
   - Use environment files for different deployments
   - Validate configuration at startup
   - Document provider-specific settings

2. **Resource Management**
   - Monitor memory and CPU usage
   - Set appropriate timeouts
   - Use health checks

3. **Error Handling**
   - Implement graceful degradation
   - Log errors with sufficient context
   - Set up alerting for critical failures

## Getting Help

### Self-Service Debugging

1. **Run diagnostic commands** from the Quick Diagnostic section
2. **Check logs** for specific error messages
3. **Test components individually** (AI provider, database, MCP server)
4. **Review configuration** against the Configuration Guide

### Community Support

1. **Search existing issues** in the project repository
2. **Check troubleshooting documentation** (this guide)
3. **Review provider-specific documentation** (OpenAI, Ollama)

### Creating Support Requests

When creating an issue, include:

```bash
# System information
./scripts/health_check.py --provider your_provider --env your_env --output debug.json

# Configuration (remove sensitive data)
env | grep -E "(AI_PROVIDER|OPENAI|OLLAMA)" | sed 's/API_KEY=.*/API_KEY=***/' | sort

# Error logs
docker compose logs your_service --tail=50

# Version information
docker --version
python --version
pip list | grep -E "(openai|ollama|crawl4ai)"
```

This troubleshooting guide covers the most common issues and their solutions. For complex or persistent issues, use the diagnostic commands and error code reference to gather information before seeking additional help.