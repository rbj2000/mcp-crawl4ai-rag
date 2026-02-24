# Crawl4AI MCP Server Setup for RTX 3090 vLLM

This guide shows you how to configure your Crawl4AI MCP server to work with your 5x RTX 3090 vLLM setup.

## 🎯 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │    │  Crawl4AI MCP   │    │   vLLM Servers   │
│   (Claude, etc) │◄──►│     Server      │◄──►│  (5x RTX 3090)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Vector Database │
                       │ (Supabase/SQLite)│
                       └─────────────────┘
```

## 🚀 Quick Setup

### 1. Start Your vLLM Servers

First, make sure your vLLM servers are running:

```bash
# Start all vLLM services
./start_vllm_rtx3090.sh

# Verify services are running
curl http://localhost:8000/v1/models  # Text LLM
curl http://localhost:8001/v1/models  # Embedding
curl http://localhost:8002/v1/models  # Vision
```

### 2. Configure Environment

Copy the RTX 3090 configuration:

```bash
# Copy the configuration file
cp env.rtx3090 .env

# Edit the configuration for your setup
nano .env
```

### 3. Update Configuration

Edit `.env` with your specific settings:

```bash
# Update these values in .env

# Supabase Configuration (if using Supabase)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here

# Or use SQLite for development
# VECTOR_DB_PROVIDER=sqlite
# SQLITE_DB_PATH=/app/data/vector_db.sqlite
```

### 4. Start MCP Server

```bash
# Start the MCP server
python src/crawl4ai_mcp.py

# Or with specific environment
python -m src.crawl4ai_mcp
```

## 📋 Configuration Details

### AI Provider Configuration

```bash
# vLLM Configuration
AI_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY

# Models
VLLM_TEXT_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_VISION_MODEL=llava-hf/llava-v1.6-mistral-7b-hf

# Performance
VLLM_MAX_BATCH_SIZE=32
VLLM_TIMEOUT=120.0
```

### Vector Database Options

#### Option 1: Supabase (Production)
```bash
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
EMBEDDING_DIMENSION=1024
```

#### Option 2: SQLite (Development)
```bash
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/app/data/vector_db.sqlite
EMBEDDING_DIMENSION=1024
```

### RAG Strategy Configuration

```bash
# Enhanced RAG Features
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false
```

## 🔧 Server Configuration

### MCP Server Settings
```bash
HOST=0.0.0.0
PORT=8051
TRANSPORT_MODE=sse
```

### Optional: Neo4j Knowledge Graph
```bash
# Enable knowledge graph features
USE_KNOWLEDGE_GRAPH=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## 🧪 Testing Your Setup

### 1. Health Check
```bash
# Test MCP server health
curl -X POST http://localhost:8051/health_check
```

### 2. Test vLLM Connectivity
```bash
# Test text generation
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Test embeddings
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "test embedding"
  }'

# Test vision
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/llava-v1.6-mistral-7b-hf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

### 3. Test MCP Tools
```bash
# Test crawling
curl -X POST http://localhost:8051/crawl_single_page \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test RAG query
curl -X POST http://localhost:8051/perform_rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "match_count": 5}'
```

## 📊 Performance Monitoring

### GPU Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor specific GPUs
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### Container Monitoring
```bash
# Monitor vLLM containers
docker stats vllm-text-server vllm-embedding-server vllm-vision-server

# View logs
docker-compose -f docker-compose.vllm.yml logs -f
```

### MCP Server Monitoring
```bash
# Monitor MCP server
ps aux | grep crawl4ai_mcp
netstat -tlnp | grep 8051
```

## 🔍 Troubleshooting

### Common Issues

#### 1. vLLM Connection Failed
```bash
# Check if vLLM servers are running
docker ps | grep vllm

# Check logs
docker logs vllm-text-server
docker logs vllm-embedding-server
docker logs vllm-vision-server

# Restart if needed
docker-compose -f docker-compose.vllm.yml restart
```

#### 2. MCP Server Won't Start
```bash
# Check environment variables
env | grep -E "(VLLM|SUPABASE|AI_PROVIDER)"

# Check Python dependencies
pip list | grep -E "(mcp|crawl4ai|supabase)"

# Check logs
python src/crawl4ai_mcp.py 2>&1 | tee mcp_server.log
```

#### 3. Database Connection Issues
```bash
# Test Supabase connection
python -c "
from supabase import create_client
import os
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')
client = create_client(url, key)
print('Supabase connection:', client.table('crawled_pages').select('*').limit(1).execute())
"

# Test SQLite
python -c "
import sqlite3
import os
db_path = os.getenv('SQLITE_DB_PATH', './local_test.db')
conn = sqlite3.connect(db_path)
print('SQLite connection: OK')
conn.close()
"
```

### Performance Issues

#### High GPU Memory Usage
```bash
# Check GPU memory
nvidia-smi

# Reduce batch sizes in docker-compose.vllm.yml
# Change --max-num-batched-tokens and --max-num-seqs
```

#### Slow Response Times
```bash
# Check network latency
ping localhost

# Monitor CPU usage
htop

# Check disk I/O
iostat -x 1
```

## 🎯 Expected Performance

### With 5x RTX 3090 Setup
- **Text Generation**: 50-100 tokens/second
- **Embeddings**: 1000-2000 texts/second
- **Vision Processing**: 10-20 images/second
- **Crawling**: 10-50 pages/minute
- **RAG Queries**: 1-3 seconds per query

### Memory Usage
- **Text LLM**: ~40GB VRAM (2x GPUs)
- **Embedding**: ~20GB VRAM (1x GPU)
- **Vision**: ~40GB VRAM (2x GPUs)
- **System RAM**: ~50-100GB
- **MCP Server**: ~2-4GB RAM

## 🔄 Maintenance

### Regular Tasks
```bash
# Update vLLM images
docker pull vllm/vllm-openai:latest
docker-compose -f docker-compose.vllm.yml up -d

# Clean up logs
docker system prune -f

# Backup database
# For Supabase: Use Supabase dashboard
# For SQLite: cp vector_db.sqlite backup_$(date +%Y%m%d).sqlite
```

### Monitoring Scripts
```bash
# Create monitoring script
cat > monitor_setup.sh << 'EOF'
#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo -e "\n=== Container Status ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n=== MCP Server Status ==="
curl -s http://localhost:8051/health_check | jq .

echo -e "\n=== vLLM Endpoints ==="
curl -s http://localhost:8000/v1/models | jq '.data[0].id' 2>/dev/null || echo "Text LLM: DOWN"
curl -s http://localhost:8001/v1/models | jq '.data[0].id' 2>/dev/null || echo "Embedding: DOWN"
curl -s http://localhost:8002/v1/models | jq '.data[0].id' 2>/dev/null || echo "Vision: DOWN"
EOF

chmod +x monitor_setup.sh
./monitor_setup.sh
```

## 🎉 Success Indicators

Your setup is working correctly when:

1. ✅ All vLLM containers are running
2. ✅ MCP server starts without errors
3. ✅ Health check returns "healthy" status
4. ✅ GPU utilization is 70-90% during processing
5. ✅ RAG queries return relevant results
6. ✅ Crawling operations complete successfully

## 📞 Support

If you encounter issues:

1. Check the logs: `docker-compose -f docker-compose.vllm.yml logs`
2. Verify GPU status: `nvidia-smi`
3. Test individual components: Use the test commands above
4. Check environment variables: `env | grep -E "(VLLM|SUPABASE)"`

Your RTX 3090 setup should provide excellent performance for the full RAG pipeline with multimodal capabilities!
