#!/bin/bash

# Crawl4AI MCP Server Startup Script for RTX 3090 vLLM Setup
# This script starts the MCP server configured for your 5x RTX 3090 vLLM setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
ENV_FILE="env.rtx3090"
MCP_SERVER="src/crawl4ai_mcp.py"
PYTHON_PATH="/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag"

echo -e "${PURPLE}🚀 Starting Crawl4AI MCP Server for RTX 3090 Setup${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Environment file $ENV_FILE not found${NC}"
    echo -e "${YELLOW}   Please create the environment file first${NC}"
    exit 1
fi

if [ ! -f "$MCP_SERVER" ]; then
    echo -e "${RED}❌ MCP server file $MCP_SERVER not found${NC}"
    exit 1
fi

# Check if vLLM servers are running
echo -e "${BLUE}🔍 Checking vLLM server connectivity...${NC}"

check_vllm_service() {
    local service_name=$1
    local port=$2
    local max_attempts=5
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is running on port $port${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - $service_name not ready...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}❌ $service_name failed to respond on port $port${NC}"
    return 1
}

# Check all vLLM services
TEXT_STATUS="❌ Failed"
EMBEDDING_STATUS="❌ Failed"
VISION_STATUS="❌ Failed"

if check_vllm_service "Text LLM" "8000"; then
    TEXT_STATUS="✅ Ready"
fi

if check_vllm_service "Embedding" "8001"; then
    EMBEDDING_STATUS="✅ Ready"
fi

if check_vllm_service "Vision" "8002"; then
    VISION_STATUS="✅ Ready"
fi

# Display vLLM status
echo -e "\n${PURPLE}📊 vLLM Server Status:${NC}"
echo -e "  Text LLM (Port 8000):     $TEXT_STATUS"
echo -e "  Embedding (Port 8001):   $EMBEDDING_STATUS"
echo -e "  Vision (Port 8002):      $VISION_STATUS"

# Check if all services are ready
if [[ "$TEXT_STATUS" == *"Failed"* ]] || [[ "$EMBEDDING_STATUS" == *"Failed"* ]]; then
    echo -e "\n${RED}❌ Required vLLM services are not running${NC}"
    echo -e "${YELLOW}   Please start your vLLM servers first:${NC}"
    echo -e "${YELLOW}   ./start_vllm_rtx3090.sh${NC}"
    exit 1
fi

# Set up environment
echo -e "\n${BLUE}🔧 Setting up environment...${NC}"

# Copy environment file
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" .env
    echo -e "${GREEN}✅ Environment file copied${NC}"
else
    echo -e "${RED}❌ Environment file $ENV_FILE not found${NC}"
    exit 1
fi

# Set Python path
export PYTHONPATH="$PYTHON_PATH:$PYTHONPATH"
echo -e "${GREEN}✅ Python path set${NC}"

# Check Python dependencies
echo -e "\n${BLUE}🔍 Checking Python dependencies...${NC}"

required_packages=("mcp" "crawl4ai" "supabase" "requests" "aiohttp")
missing_packages=()

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✅ $package${NC}"
    else
        echo -e "${RED}❌ $package${NC}"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "\n${YELLOW}⚠️  Missing packages: ${missing_packages[*]}${NC}"
    echo -e "${YELLOW}   Install with: pip install ${missing_packages[*]}${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test database connectivity
echo -e "\n${BLUE}🔍 Testing database connectivity...${NC}"

# Check if Supabase is configured
if grep -q "VECTOR_DB_PROVIDER=supabase" .env; then
    SUPABASE_URL=$(grep "SUPABASE_URL=" .env | cut -d'=' -f2)
    SUPABASE_KEY=$(grep "SUPABASE_SERVICE_KEY=" .env | cut -d'=' -f2)
    
    if [ "$SUPABASE_URL" != "https://your-project.supabase.co" ] && [ "$SUPABASE_KEY" != "your-service-key-here" ]; then
        echo -e "${GREEN}✅ Supabase configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Supabase not configured - using default values${NC}"
    fi
elif grep -q "VECTOR_DB_PROVIDER=sqlite" .env; then
    echo -e "${GREEN}✅ SQLite configured${NC}"
else
    echo -e "${YELLOW}⚠️  No database provider configured${NC}"
fi

# Display configuration summary
echo -e "\n${PURPLE}📋 Configuration Summary:${NC}"
echo -e "  AI Provider: $(grep "AI_PROVIDER=" .env | cut -d'=' -f2)"
echo -e "  vLLM URL: $(grep "VLLM_BASE_URL=" .env | cut -d'=' -f2)"
echo -e "  Database: $(grep "VECTOR_DB_PROVIDER=" .env | cut -d'=' -f2)"
echo -e "  MCP Port: $(grep "PORT=" .env | cut -d'=' -f2)"
echo -e "  Transport: $(grep "TRANSPORT_MODE=" .env | cut -d'=' -f2)"

# Start MCP server
echo -e "\n${BLUE}🚀 Starting MCP server...${NC}"
echo -e "${PURPLE}   Server will be available at: http://localhost:8051${NC}"
echo -e "${PURPLE}   Press Ctrl+C to stop the server${NC}"

# Create logs directory
mkdir -p logs

# Start the server with logging
echo -e "\n${GREEN}🎉 MCP Server starting...${NC}"
echo -e "${BLUE}📝 Logs will be written to: logs/mcp_server.log${NC}"

# Run the MCP server
cd "$PYTHON_PATH"
python3 "$MCP_SERVER" 2>&1 | tee logs/mcp_server.log
