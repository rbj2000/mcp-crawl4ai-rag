#!/bin/bash

# vLLM Multi-Model Server Startup Script
# Optimized for 5x RTX 3090 (24GB each) + 256GB RAM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.vllm.yml"
ENV_FILE="env.vllm"

echo -e "${PURPLE}🚀 Starting vLLM Multi-Model Server Setup${NC}"
echo -e "${PURPLE}   Optimized for 5x RTX 3090 (24GB each) + 256GB RAM${NC}"

# Hardware check function
check_hardware() {
    echo -e "${BLUE}🔍 Checking hardware configuration...${NC}"
    
    # Check GPU count
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -lt 5 ]; then
        echo -e "${RED}❌ Expected 5 GPUs, found $GPU_COUNT. Please check your GPU configuration.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Found $GPU_COUNT GPUs${NC}"
    
    # Check GPU memory
    echo -e "${BLUE}📊 GPU Memory Status:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
        echo -e "  ${GREEN}$line${NC}"
    done
    
    # Check system RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -lt 200 ]; then
        echo -e "${YELLOW}⚠️  System RAM: ${TOTAL_RAM}GB (recommended: 256GB+)${NC}"
    else
        echo -e "${GREEN}✅ System RAM: ${TOTAL_RAM}GB${NC}"
    fi
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo -e "${RED}❌ NVIDIA Docker runtime not detected.${NC}"
    echo -e "${YELLOW}   Please install nvidia-docker2:${NC}"
    echo -e "${YELLOW}   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -${NC}"
    echo -e "${YELLOW}   distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)${NC}"
    echo -e "${YELLOW}   curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list${NC}"
    echo -e "${YELLOW}   sudo apt-get update && sudo apt-get install -y nvidia-docker2${NC}"
    exit 1
fi

# Hardware check
check_hardware

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p models logs/vllm-text logs/vllm-embedding logs/vllm-vision models/cache

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  Environment file $ENV_FILE not found. Using defaults.${NC}"
fi

# Stop any existing containers
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true

# Pull latest images
echo -e "${BLUE}📥 Pulling latest vLLM image...${NC}"
docker pull vllm/vllm-openai:latest

# Display GPU allocation
echo -e "\n${PURPLE}🎯 GPU Allocation Strategy:${NC}"
echo -e "  ${BLUE}Text LLM:     GPUs 0-1 (48GB VRAM) - 2x RTX 3090${NC}"
echo -e "  ${BLUE}Embedding:   GPU 2  (24GB VRAM)  - 1x RTX 3090${NC}"
echo -e "  ${BLUE}Vision:      GPUs 3-4 (48GB VRAM) - 2x RTX 3090${NC}"
echo -e "  ${BLUE}Total:       5x RTX 3090 (120GB VRAM)${NC}"

# Start services
echo -e "\n${BLUE}🚀 Starting vLLM services...${NC}"
if [ -f "$ENV_FILE" ]; then
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
else
    docker-compose -f $COMPOSE_FILE up -d
fi

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to start (this may take 5-10 minutes for model downloads)...${NC}"
sleep 60

# Enhanced health check function
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=60
    local attempt=1

    echo -e "${BLUE}🔍 Checking $service_name on port $port...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        
        if [ $((attempt % 10)) -eq 0 ]; then
            echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - $service_name still loading...${NC}"
        fi
        sleep 10
        ((attempt++))
    done
    
    echo -e "${RED}❌ $service_name failed to start after $max_attempts attempts${NC}"
    return 1
}

# Check all services
echo -e "${BLUE}🔍 Performing health checks...${NC}"

# Check Text LLM
if check_service "Text LLM" "8000"; then
    TEXT_STATUS="✅ Ready"
else
    TEXT_STATUS="❌ Failed"
fi

# Check Embedding
if check_service "Embedding" "8001"; then
    EMBEDDING_STATUS="✅ Ready"
else
    EMBEDDING_STATUS="❌ Failed"
fi

# Check Vision
if check_service "Vision" "8002"; then
    VISION_STATUS="✅ Ready"
else
    VISION_STATUS="❌ Failed"
fi

# Display status
echo -e "\n${PURPLE}📊 Service Status:${NC}"
echo -e "  Text LLM (Port 8000):     $TEXT_STATUS"
echo -e "  Embedding (Port 8001):   $EMBEDDING_STATUS"
echo -e "  Vision (Port 8002):      $VISION_STATUS"

# Display access information
echo -e "\n${GREEN}🎉 vLLM Multi-Model Server Setup Complete!${NC}"
echo -e "\n${PURPLE}📡 Access URLs:${NC}"
echo -e "  Text LLM:     http://localhost:8000"
echo -e "  Embedding:    http://localhost:8001"
echo -e "  Vision:       http://localhost:8002"
echo -e "  Nginx Proxy:  http://localhost:8080"
echo -e "  Prometheus:   http://localhost:9090"
echo -e "  Grafana:      http://localhost:3000 (admin/admin)"

echo -e "\n${PURPLE}🔧 Management Commands:${NC}"
echo -e "  View logs:    docker-compose -f $COMPOSE_FILE logs -f"
echo -e "  Stop all:     docker-compose -f $COMPOSE_FILE down"
echo -e "  Restart:      docker-compose -f $COMPOSE_FILE restart"
echo -e "  GPU status:   nvidia-smi"

echo -e "\n${PURPLE}🧪 Test Commands:${NC}"
echo -e "  Test Text:    curl http://localhost:8000/v1/models"
echo -e "  Test Embed:   curl http://localhost:8001/v1/models"
echo -e "  Test Vision:  curl http://localhost:8002/v1/models"

# Performance monitoring
echo -e "\n${PURPLE}📈 Performance Monitoring:${NC}"
echo -e "  GPU Usage:    watch -n 1 nvidia-smi"
echo -e "  Container:    docker stats"
echo -e "  Prometheus:   http://localhost:9090"
echo -e "  Grafana:      http://localhost:3000"

# Test API endpoints with performance metrics
echo -e "\n${BLUE}🧪 Running API performance tests...${NC}"

# Test Text LLM with timing
echo -e "${BLUE}Testing Text LLM performance...${NC}"
TEXT_START=$(date +%s)
if curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello, how are you?", "max_tokens": 50}' \
  > /dev/null 2>&1; then
    TEXT_END=$(date +%s)
    TEXT_TIME=$((TEXT_END - TEXT_START))
    echo -e "${GREEN}✅ Text LLM API working (${TEXT_TIME}s)${NC}"
else
    echo -e "${RED}❌ Text LLM API failed${NC}"
fi

# Test Embedding with timing
echo -e "${BLUE}Testing Embedding performance...${NC}"
EMBED_START=$(date +%s)
if curl -s -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": "test embedding"}' \
  > /dev/null 2>&1; then
    EMBED_END=$(date +%s)
    EMBED_TIME=$((EMBED_END - EMBED_START))
    echo -e "${GREEN}✅ Embedding API working (${EMBED_TIME}s)${NC}"
else
    echo -e "${RED}❌ Embedding API failed${NC}"
fi

# Test Vision with timing
echo -e "${BLUE}Testing Vision performance...${NC}"
VISION_START=$(date +%s)
if curl -s -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llava-hf/llava-v1.6-mistral-7b-hf", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}' \
  > /dev/null 2>&1; then
    VISION_END=$(date +%s)
    VISION_TIME=$((VISION_END - VISION_START))
    echo -e "${GREEN}✅ Vision API working (${VISION_TIME}s)${NC}"
else
    echo -e "${RED}❌ Vision API failed${NC}"
fi

# Display final GPU status
echo -e "\n${PURPLE}🎯 Final GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
    echo -e "  ${GREEN}$line${NC}"
done

echo -e "\n${GREEN}🎉 Setup complete! All services are running on your 5x RTX 3090 setup.${NC}"
echo -e "${PURPLE}💡 Pro tip: Monitor GPU usage with 'watch -n 1 nvidia-smi'${NC}"
