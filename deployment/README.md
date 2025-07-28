# MCP Crawl4AI Deployment Guide

This guide covers deployment orchestration for the MCP Crawl4AI RAG system with multi-provider AI support.

## Quick Start

### 1. Choose Your Provider Configuration

```bash
# OpenAI + Supabase (Production)
./scripts/deploy_provider.py --provider openai --database supabase --env production

# Ollama + SQLite (Development)
./scripts/deploy_provider.py --provider ollama --database sqlite --env development

# Hybrid OpenAI/Ollama + Supabase (Cost-Optimized)
./scripts/deploy_provider.py --provider hybrid --database supabase --env production
```

### 2. Health Check Your Deployment

```bash
# Check deployment health
./scripts/health_check.py --provider ollama --env development

# Continuous monitoring
./scripts/health_check.py --provider hybrid --env production --output health_report.json
```

## Supported Configurations

### AI Providers

| Provider | Description | Use Case | Requirements |
|----------|-------------|----------|--------------|
| `openai` | OpenAI GPT models with OpenAI embeddings | Production, reliability | `OPENAI_API_KEY` |
| `ollama` | Local Ollama models (LLM + embeddings) | Development, privacy | Ollama service |
| `hybrid` | OpenAI embeddings + Ollama LLM | Cost optimization | Both API key and Ollama |

### Vector Databases

| Database | Description | Use Case | Requirements |
|----------|-------------|----------|--------------|
| `sqlite` | Local SQLite with vector support | Development, testing | None |
| `supabase` | Managed PostgreSQL with pgvector | Production, scaling | Supabase credentials |
| `neo4j` | Neo4j with vector and knowledge graph | Advanced analytics | Neo4j instance |
| `pinecone` | Managed vector database | High-performance search | Pinecone API key |
| `weaviate` | Open-source vector database | Hybrid search, ML | Weaviate instance |

### Environment Profiles

| Environment | AI Provider | Database | Features | Use Case |
|-------------|-------------|----------|----------|----------|
| `development` | Configurable | SQLite | All features enabled | Local development |
| `production` | OpenAI | Supabase | Optimized for scale | Production deployment |
| `ollama` | Ollama | SQLite | Local AI stack | Privacy-focused development |
| `hybrid` | Mixed | Supabase | Cost-optimized | Production with cost control |

## Docker Compose Profiles

### Available Profiles

```bash
# List all available profiles
docker compose config --profiles

# OpenAI + Supabase
docker compose --profile supabase up -d

# Ollama + SQLite (local development)
docker compose --profile ollama-sqlite up -d

# Ollama + Neo4j (full AI stack)
docker compose --profile ollama-full up -d

# Hybrid deployment
docker compose --profile hybrid up -d

# With monitoring
docker compose --profile ollama-full --profile monitoring up -d
```

### Profile Combinations

| Profile | Services | Ports | Use Case |
|---------|----------|-------|----------|
| `supabase` | MCP + Supabase connection | 8051 | Production with managed DB |
| `sqlite` | MCP + SQLite | 8052 | Local development |
| `ollama` | MCP + Ollama + SQLite | 8056, 11434 | Local AI development |
| `ollama-full` | MCP + Ollama + Neo4j + Monitoring | 8057, 11434, 7474, 9090, 3000 | Complete local stack |
| `hybrid` | MCP + Ollama + Supabase | 8058, 11434 | Cost-optimized production |
| `neo4j` | MCP + Neo4j | 8055, 7474, 7687 | Knowledge graph analytics |
| `monitoring` | Prometheus + Grafana | 9090, 3000 | Monitoring stack |

## Environment Configuration

### Environment Files

Environment-specific configurations are stored in `environments/`:

- `.env.development` - Local development with hot reload
- `.env.production` - Production with security and performance optimizations
- `.env.ollama` - Ollama-focused configuration
- `.env.hybrid` - Mixed provider configuration

### Key Environment Variables

#### AI Provider Configuration
```bash
# Provider Selection
AI_PROVIDER=openai|ollama|mixed
EMBEDDING_PROVIDER=openai|ollama
LLM_PROVIDER=openai|ollama

# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
MODEL_CHOICE=gpt-4o-mini

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSION=768
```

#### Database Configuration
```bash
# Vector Database Selection
VECTOR_DB_PROVIDER=sqlite|supabase|neo4j_vector|pinecone|weaviate

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# SQLite
SQLITE_DB_PATH=/app/data/vector_db.sqlite
```

#### RAG Strategy Configuration
```bash
USE_CONTEXTUAL_EMBEDDINGS=true|false
USE_HYBRID_SEARCH=true|false
USE_AGENTIC_RAG=true|false
USE_RERANKING=true|false
USE_KNOWLEDGE_GRAPH=true|false
```

## Deployment Scripts

### deploy_provider.py

Main deployment orchestration script:

```bash
# Deploy with specific configuration
./scripts/deploy_provider.py \
  --provider ollama \
  --database sqlite \
  --env development \
  --build

# Deploy in foreground with logs
./scripts/deploy_provider.py \
  --provider hybrid \
  --database supabase \
  --env production \
  --foreground

# List available configurations
./scripts/deploy_provider.py --list-profiles

# Stop all services
./scripts/deploy_provider.py --stop

# View logs
./scripts/deploy_provider.py --logs
./scripts/deploy_provider.py --logs mcp-crawl4ai-ollama-sqlite --follow
```

### health_check.py

Comprehensive health monitoring:

```bash
# Basic health check
./scripts/health_check.py --provider ollama --env development

# Detailed check with JSON export
./scripts/health_check.py \
  --provider hybrid \
  --env production \
  --output health_report.json

# Simulation mode for testing
./scripts/health_check.py --simulate
```

### setup_ollama.py

Ollama management and setup:

```bash
# Install Ollama
./scripts/setup_ollama.py --install

# Pull required models
./scripts/setup_ollama.py --pull-models

# Include optional models
./scripts/setup_ollama.py --pull-models --include-optional

# Validate model functionality
./scripts/setup_ollama.py --validate

# Apply performance optimization
./scripts/setup_ollama.py --optimize balanced

# Show comprehensive status
./scripts/setup_ollama.py --status
```

## Health Checks and Monitoring

### Built-in Health Checks

All Docker containers include health checks:

```bash
# View container health
docker compose ps

# Check health endpoint
curl http://localhost:8051/health

# Ollama health
curl http://localhost:11434/api/tags
```

### Monitoring Stack

When using the `monitoring` profile:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

Pre-configured dashboards include:
- MCP server metrics
- AI provider performance
- Database connection health
- Resource utilization

### Health Check Features

The health check script validates:

- ✅ Service endpoints (HTTP/TCP)
- ✅ Docker container status
- ✅ AI model availability
- ✅ Database connectivity
- ✅ System resource usage
- ✅ End-to-end functionality

## Ollama Setup

### Automatic Installation

```bash
# Full Ollama setup
./scripts/setup_ollama.py --install --pull-models --optimize balanced
```

### Manual Setup

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Start Service**:
   ```bash
   ollama serve
   ```

3. **Pull Models**:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
   ```

### Performance Optimization

Choose optimization profile based on your system:

```bash
# Low memory systems (< 8GB RAM)
./scripts/setup_ollama.py --optimize low_memory

# Balanced systems (8-16GB RAM)
./scripts/setup_ollama.py --optimize balanced

# High-performance systems (> 16GB RAM)
./scripts/setup_ollama.py --optimize high_performance
```

## Common Deployment Scenarios

### Local Development (Ollama)

```bash
# Setup Ollama
./scripts/setup_ollama.py --install --pull-models

# Deploy with SQLite
./scripts/deploy_provider.py --provider ollama --database sqlite --env development

# Health check
./scripts/health_check.py --provider ollama --env development
```

### Production (OpenAI + Supabase)

```bash
# Set environment variables
export OPENAI_API_KEY=your_key
export SUPABASE_URL=your_url
export SUPABASE_SERVICE_KEY=your_key

# Deploy
./scripts/deploy_provider.py --provider openai --database supabase --env production --build

# Health check
./scripts/health_check.py --provider openai --env production
```

### Cost-Optimized (Hybrid)

```bash
# Setup Ollama for LLM
./scripts/setup_ollama.py --install --pull-models

# Set OpenAI key for embeddings
export OPENAI_API_KEY=your_key
export SUPABASE_URL=your_url
export SUPABASE_SERVICE_KEY=your_key

# Deploy hybrid stack
./scripts/deploy_provider.py --provider hybrid --database supabase --env hybrid

# Monitor health
./scripts/health_check.py --provider hybrid --env hybrid
```

### Knowledge Graph Analytics (Neo4j)

```bash
# Deploy with Neo4j
./scripts/deploy_provider.py --provider openai --database neo4j --env production

# Access Neo4j browser
open http://localhost:7474
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   ./scripts/setup_ollama.py --status
   
   # Start Ollama service
   ./scripts/setup_ollama.py --start
   ```

2. **Models Not Available**
   ```bash
   # Pull missing models
   ./scripts/setup_ollama.py --pull-models
   
   # Validate models
   ./scripts/setup_ollama.py --validate
   ```

3. **Container Health Checks Failing**
   ```bash
   # Check container logs
   ./scripts/deploy_provider.py --logs
   
   # Run health check
   ./scripts/health_check.py --provider your_provider --env your_env
   ```

4. **Memory Issues with Ollama**
   ```bash
   # Apply low memory optimization
   ./scripts/setup_ollama.py --optimize low_memory
   
   # Restart services
   ./scripts/deploy_provider.py --stop
   ./scripts/deploy_provider.py --provider ollama --database sqlite --env development
   ```

### Logs and Debugging

```bash
# Container logs
docker compose logs -f mcp-crawl4ai-ollama-sqlite

# Health check with details
./scripts/health_check.py --provider ollama --env development --output debug.json

# Ollama logs
docker compose logs -f ollama

# System resource check
docker stats
```

### Performance Tuning

1. **Ollama Performance**:
   - Use `--optimize` based on your system specs
   - Monitor memory usage with `docker stats`
   - Adjust `OLLAMA_NUM_PARALLEL` for concurrency

2. **Database Performance**:
   - Use appropriate vector dimensions
   - Enable connection pooling for production
   - Monitor query performance

3. **Container Resources**:
   - Set resource limits in docker-compose.yml
   - Use health checks to detect resource exhaustion
   - Scale services based on load

## Security Considerations

### Production Deployment

1. **API Keys**: Use environment files, never commit keys
2. **Network Security**: Configure CORS appropriately
3. **Container Security**: Run as non-root user (configured)
4. **Database Security**: Use service roles, not admin keys
5. **Monitoring**: Enable security monitoring and alerting

### Environment Isolation

```bash
# Use separate environment files
cp deployment/environments/.env.production.example .env.production
# Edit with your production values

# Deploy with environment file
./scripts/deploy_provider.py --env production
```

## Support and Maintenance

### Regular Maintenance

```bash
# Update models
./scripts/setup_ollama.py --pull-models

# Health checks
./scripts/health_check.py --provider your_provider --env your_env

# Container cleanup
docker system prune -f

# Log rotation
docker compose logs --since=24h > logs/$(date +%Y%m%d).log
```

### Backup and Recovery

```bash
# Backup SQLite database
cp data/vector_db.sqlite backups/vector_db_$(date +%Y%m%d).sqlite

# Export Supabase data (use Supabase CLI)
supabase db dump -f backup.sql

# Container volume backup
docker run --rm -v mcp_data:/data -v $(pwd)/backups:/backup alpine tar czf /backup/data_backup.tar.gz /data
```

For additional support, check the project documentation and GitHub issues.