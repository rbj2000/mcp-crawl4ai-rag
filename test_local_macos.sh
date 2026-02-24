#!/bin/bash

# Local Testing Script for Crawl4AI MCP Server on macOS
# This script sets up and tests the MCP server locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🧪 Crawl4AI MCP Server Local Testing on macOS${NC}"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This script is designed for macOS${NC}"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    echo -e "${YELLOW}   Install with: brew install python3${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 found: $(python3 --version)${NC}"

# Check if we're in the right directory
if [ ! -f "src/crawl4ai_mcp.py" ]; then
    echo -e "${RED}❌ MCP server file not found${NC}"
    echo -e "${YELLOW}   Please run this script from the project root directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ MCP server file found${NC}"

# Create local testing environment
echo -e "\n${BLUE}🔧 Setting up local testing environment...${NC}"

cat > .env.local << 'EOF'
# Local Testing Configuration for macOS

# =============================================================================
# AI Provider Configuration (Ollama for local testing)
# =============================================================================
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_DIMENSIONS=768

# =============================================================================
# Vector Database Configuration (SQLite)
# =============================================================================
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=./local_test.db
EMBEDDING_DIMENSION=768

# =============================================================================
# RAG Strategy Configuration
# =============================================================================
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# =============================================================================
# Server Configuration
# =============================================================================
HOST=0.0.0.0
PORT=8051
TRANSPORT_MODE=sse

# =============================================================================
# Performance Settings
# =============================================================================
OLLAMA_MAX_BATCH_SIZE=10
OLLAMA_TIMEOUT=120.0

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO
PYTHONPATH=/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag
EOF

echo -e "${GREEN}✅ Local environment file created${NC}"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not found. Installing Ollama...${NC}"
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Ollama installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install Ollama${NC}"
        echo -e "${YELLOW}   Please install manually from: https://ollama.ai/download${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama found: $(ollama --version)${NC}"
fi

# Start Ollama service
echo -e "\n${BLUE}🚀 Starting Ollama service...${NC}"

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ollama is already running${NC}"
else
    echo -e "${BLUE}Starting Ollama service...${NC}"
    ollama serve &
    OLLAMA_PID=$!
    echo -e "${GREEN}✅ Ollama started with PID: $OLLAMA_PID${NC}"
    
    # Wait for Ollama to start
    echo -e "${BLUE}⏳ Waiting for Ollama to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Ollama is ready${NC}"
            break
        fi
        echo -e "${YELLOW}⏳ Attempt $i/30 - Ollama starting...${NC}"
        sleep 2
    done
fi

# Pull required models
echo -e "\n${BLUE}📥 Pulling required models...${NC}"

# Check if models are already available
if ollama list | grep -q "llama3.2:1b"; then
    echo -e "${GREEN}✅ Llama model already available${NC}"
else
    echo -e "${BLUE}Downloading Llama model (this may take a few minutes)...${NC}"
    ollama pull llama3.2:1b
fi

if ollama list | grep -q "nomic-embed-text"; then
    echo -e "${GREEN}✅ Embedding model already available${NC}"
else
    echo -e "${BLUE}Downloading embedding model...${NC}"
    ollama pull nomic-embed-text
fi

# Test Ollama connectivity
echo -e "\n${BLUE}🔍 Testing Ollama connectivity...${NC}"

# Test embedding
if curl -s -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "prompt": "test"}' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Embedding model working${NC}"
else
    echo -e "${RED}❌ Embedding model test failed${NC}"
fi

# Test completion
if curl -s -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "prompt": "Hello", "stream": false}' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ LLM model working${NC}"
else
    echo -e "${RED}❌ LLM model test failed${NC}"
fi

# Install Python dependencies
echo -e "\n${BLUE}📦 Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed from requirements.txt${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, installing basic dependencies...${NC}"
    pip3 install mcp crawl4ai supabase requests aiohttp
fi

# Create logs directory
mkdir -p logs

# Copy environment file
cp .env.local .env

echo -e "\n${GREEN}🎉 Setup complete! Starting MCP server...${NC}"
echo -e "${PURPLE}   Server will be available at: http://localhost:8051${NC}"
echo -e "${PURPLE}   Press Ctrl+C to stop the server${NC}"

# Start MCP server
echo -e "\n${BLUE}🚀 Starting MCP server...${NC}"
python3 src/crawl4ai_mcp.py
