# vLLM Multi-Model Server Setup for 5x RTX 3090

This setup optimizes vLLM deployment for your hardware configuration: **5x RTX 3090 (24GB each) + 256GB RAM**.

## 🎯 Hardware Configuration

### GPU Allocation Strategy
- **Text LLM**: GPUs 0-1 (48GB VRAM) - 2x RTX 3090
- **Embedding**: GPU 2 (24GB VRAM) - 1x RTX 3090  
- **Vision**: GPUs 3-4 (48GB VRAM) - 2x RTX 3090
- **Total**: 5x RTX 3090 (120GB VRAM)

### System Requirements
- **GPUs**: 5x RTX 3090 (24GB each)
- **RAM**: 256GB system memory
- **Storage**: 500GB+ NVMe SSD (for models)
- **CPU**: 16+ cores recommended

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install NVIDIA drivers and CUDA
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Install Docker and NVIDIA Docker runtime
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Start All Services
```bash
# Make script executable
chmod +x start_vllm_rtx3090.sh

# Start all vLLM services
./start_vllm_rtx3090.sh
```

### 3. Verify Setup
```bash
# Check GPU allocation
nvidia-smi

# Test all endpoints
curl http://localhost:8000/v1/models  # Text LLM
curl http://localhost:8001/v1/models  # Embedding
curl http://localhost:8002/v1/models  # Vision
```

## 📊 Service Endpoints

| Service | Port | GPUs | Purpose |
|---------|------|------|---------|
| Text LLM | 8000 | 0-1 | Text generation, chat completions |
| Embedding | 8001 | 2 | Vector embeddings for RAG |
| Vision | 8002 | 3-4 | Image understanding, multimodal |
| Nginx Proxy | 8080 | - | Load balancer and routing |
| Prometheus | 9090 | - | Metrics collection |
| Grafana | 3000 | - | Monitoring dashboard |

## 🔧 Configuration Files

### Docker Compose (`docker-compose.vllm.yml`)
- **Text LLM**: 2x GPUs, tensor parallelism, 16K context
- **Embedding**: 1x GPU, optimized for batch processing
- **Vision**: 2x GPUs, tensor parallelism for image processing

### Environment (`env.vllm`)
- GPU allocation per service
- Performance tuning parameters
- Memory optimization settings

### Nginx (`nginx.conf`)
- Load balancing configuration
- Path-based routing
- Health check endpoints

## 🧪 Testing Your Setup

### Basic API Tests
```bash
# Test Text LLM
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# Test Embedding
curl -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-en-v1.5",
    "input": "test embedding"
  }'

# Test Vision
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llava-hf/llava-v1.6-mistral-7b-hf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }'
```

### Performance Tests
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor container resources
docker stats

# Check service health
curl http://localhost:8080/health
```

## 📈 Performance Optimization

### GPU Memory Utilization
- **Text LLM**: 85% GPU memory (20.4GB per GPU)
- **Embedding**: 80% GPU memory (19.2GB)
- **Vision**: 80% GPU memory (19.2GB per GPU)

### Batch Processing
- **Text LLM**: 256 concurrent sequences, 8K tokens
- **Embedding**: 512 concurrent sequences, 16K tokens
- **Vision**: 128 concurrent sequences, 4K tokens

### Tensor Parallelism
- **Text LLM**: 2-way tensor parallelism across 2 GPUs
- **Vision**: 2-way tensor parallelism across 2 GPUs
- **Embedding**: Single GPU (no parallelism needed)

## 🔍 Monitoring and Debugging

### GPU Monitoring
```bash
# Real-time GPU status
nvidia-smi -l 1

# Detailed GPU info
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
```

### Container Monitoring
```bash
# View all container logs
docker-compose -f docker-compose.vllm.yml logs -f

# View specific service logs
docker-compose -f docker-compose.vllm.yml logs -f vllm-text
docker-compose -f docker-compose.vllm.yml logs -f vllm-embedding
docker-compose -f docker-compose.vllm.yml logs -f vllm-vision
```

### Prometheus Metrics
- Access: http://localhost:9090
- Metrics: GPU utilization, memory usage, request rates
- Alerts: High GPU temperature, memory usage

### Grafana Dashboard
- Access: http://localhost:3000
- Login: admin/admin
- Dashboards: GPU metrics, container stats, API performance

## 🛠️ Management Commands

### Start/Stop Services
```bash
# Start all services
docker-compose -f docker-compose.vllm.yml up -d

# Stop all services
docker-compose -f docker-compose.vllm.yml down

# Restart specific service
docker-compose -f docker-compose.vllm.yml restart vllm-text
```

### Update Models
```bash
# Pull latest vLLM image
docker pull vllm/vllm-openai:latest

# Restart with new image
docker-compose -f docker-compose.vllm.yml up -d --force-recreate
```

### Backup and Recovery
```bash
# Backup model cache
tar -czf models_backup.tar.gz models/

# Backup logs
tar -czf logs_backup.tar.gz logs/
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Memory Errors
```bash
# Check GPU memory usage
nvidia-smi

# Reduce GPU memory utilization in docker-compose.vllm.yml
# Change --gpu-memory-utilization from 0.85 to 0.7
```

#### Model Download Failures
```bash
# Check internet connection
ping huggingface.co

# Clear model cache and retry
rm -rf models/
docker-compose -f docker-compose.vllm.yml up -d
```

#### Service Startup Failures
```bash
# Check container logs
docker-compose -f docker-compose.vllm.yml logs vllm-text

# Check GPU availability
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Performance Issues

#### Slow Response Times
- Check GPU utilization: `nvidia-smi`
- Monitor memory usage: `docker stats`
- Adjust batch sizes in configuration

#### High GPU Temperature
- Check cooling system
- Reduce GPU memory utilization
- Monitor with: `nvidia-smi -l 1`

## 📋 Configuration for MCP Crawl4AI RAG

### Environment Variables
```bash
# AI Provider
AI_PROVIDER=vllm

# Text LLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_TEXT_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Embedding Configuration
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_EMBEDDING_DIMENSIONS=1024

# Vision Configuration
VLLM_VISION_ENABLED=true
VLLM_VISION_MODEL=llava-hf/llava-v1.6-mistral-7b-hf

# Performance
VLLM_MAX_BATCH_SIZE=32
VLLM_TIMEOUT=120.0
```

### Vector Database Configuration
```bash
# Use Supabase for production
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
EMBEDDING_DIMENSION=1024

# Or use SQLite for development
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/app/data/vector_db.sqlite
EMBEDDING_DIMENSION=1024
```

## 🎉 Expected Performance

### Throughput (RTX 3090 Optimized)
- **Text Generation**: ~50-100 tokens/second
- **Embeddings**: ~1000-2000 texts/second
- **Vision Processing**: ~10-20 images/second

### Memory Usage
- **Text LLM**: ~40GB VRAM (2x GPUs)
- **Embedding**: ~20GB VRAM (1x GPU)
- **Vision**: ~40GB VRAM (2x GPUs)
- **System RAM**: ~50-100GB

### Latency
- **Text LLM**: 1-3 seconds per request
- **Embeddings**: 0.1-0.5 seconds per batch
- **Vision**: 2-5 seconds per image

This setup provides a high-performance, scalable foundation for your RAG pipeline with full multimodal capabilities!
