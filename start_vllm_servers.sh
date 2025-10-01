#!/bin/bash

# vLLM Multi-Model Server Startup Script
# This script starts all 3 vLLM models in separate containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.vllm.yml"
ENV_FILE="env.vllm"

echo -e "${BLUE}🚀 Starting vLLM Multi-Model Server Setup${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not work.${NC}"
    echo -e "${YELLOW}   Make sure you have nvidia-docker2 installed and configured.${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p models logs/vllm-text logs/vllm-embedding logs/vllm-vision

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

# Start services
echo -e "${BLUE}🚀 Starting vLLM services...${NC}"
if [ -f "$ENV_FILE" ]; then
    docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d
else
    docker-compose -f $COMPOSE_FILE up -d
fi

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 30

# Health check function
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1

    echo -e "${BLUE}🔍 Checking $service_name on port $port...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - $service_name not ready yet...${NC}"
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
echo -e "\n${BLUE}📊 Service Status:${NC}"
echo -e "  Text LLM (Port 8000):     $TEXT_STATUS"
echo -e "  Embedding (Port 8001):   $EMBEDDING_STATUS"
echo -e "  Vision (Port 8002):      $VISION_STATUS"

# Display access information
echo -e "\n${GREEN}🎉 vLLM Multi-Model Server Setup Complete!${NC}"
echo -e "\n${BLUE}📡 Access URLs:${NC}"
echo -e "  Text LLM:     http://localhost:8000"
echo -e "  Embedding:    http://localhost:8001"
echo -e "  Vision:       http://localhost:8002"
echo -e "  Nginx Proxy:  http://localhost:8080"
echo -e "  Prometheus:   http://localhost:9090"
echo -e "  Grafana:      http://localhost:3000 (admin/admin)"

echo -e "\n${BLUE}🔧 Management Commands:${NC}"
echo -e "  View logs:    docker-compose -f $COMPOSE_FILE logs -f"
echo -e "  Stop all:     docker-compose -f $COMPOSE_FILE down"
echo -e "  Restart:      docker-compose -f $COMPOSE_FILE restart"

echo -e "\n${BLUE}🧪 Test Commands:${NC}"
echo -e "  Test Text:    curl http://localhost:8000/v1/models"
echo -e "  Test Embed:   curl http://localhost:8001/v1/models"
echo -e "  Test Vision:  curl http://localhost:8002/v1/models"

# Test API endpoints
echo -e "\n${BLUE}🧪 Running API tests...${NC}"

# Test Text LLM
echo -e "${BLUE}Testing Text LLM...${NC}"
if curl -s -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 10}' \
  > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Text LLM API working${NC}"
else
    echo -e "${RED}❌ Text LLM API failed${NC}"
fi

# Test Embedding
echo -e "${BLUE}Testing Embedding...${NC}"
if curl -s -X POST http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": "test"}' \
  > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Embedding API working${NC}"
else
    echo -e "${RED}❌ Embedding API failed${NC}"
fi

# Test Vision
echo -e "${BLUE}Testing Vision...${NC}"
if curl -s -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llava-hf/llava-v1.6-mistral-7b-hf", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}' \
  > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Vision API working${NC}"
else
    echo -e "${RED}❌ Vision API failed${NC}"
fi

echo -e "\n${GREEN}🎉 Setup complete! All services are running.${NC}"
