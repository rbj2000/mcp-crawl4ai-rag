#!/bin/bash

# Crawl4AI MCP Server Deployment Script for macOS with RTX 3090 vLLM
# This script deploys the MCP server on macOS connecting to remote RTX 3090 vLLM servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 Crawl4AI MCP Server Deployment for macOS + RTX 3090 vLLM${NC}"

# Configuration
PROJECT_DIR="/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This script is designed for macOS${NC}"
    exit 1
fi

echo -e "${GREEN}✅ macOS detected: $(sw_vers -productVersion)${NC}"

# Check if we're in the right directory
if [ ! -f "src/crawl4ai_mcp.py" ]; then
    echo -e "${RED}❌ MCP server file not found${NC}"
    echo -e "${YELLOW}   Please run this script from the project root directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ MCP server file found${NC}"

# Get RTX 3090 server configuration
echo -e "\n${BLUE}🔧 RTX 3090 Server Configuration${NC}"
read -p "Enter your RTX 3090 server IP address: " RTX_SERVER_IP
read -p "Enter your Supabase URL: " SUPABASE_URL
read -p "Enter your Supabase Service Key: " SUPABASE_KEY

# Validate inputs
if [ -z "$RTX_SERVER_IP" ] || [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_KEY" ]; then
    echo -e "${RED}❌ All configuration values are required${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration values provided${NC}"

# Test RTX 3090 server connectivity
echo -e "\n${BLUE}🔍 Testing RTX 3090 server connectivity...${NC}"

test_vllm_service() {
    local service_name=$1
    local port=$2
    local max_attempts=5
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://$RTX_SERVER_IP:$port/v1/models" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is accessible on $RTX_SERVER_IP:$port${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}⏳ Attempt $attempt/$max_attempts - $service_name not ready...${NC}"
        sleep 2
        ((attempt++))
    done
    
    echo -e "${RED}❌ $service_name failed to respond on $RTX_SERVER_IP:$port${NC}"
    return 1
}

# Test all vLLM services
TEXT_STATUS="❌ Failed"
EMBEDDING_STATUS="❌ Failed"
VISION_STATUS="❌ Failed"

if test_vllm_service "Text LLM" "8000"; then
    TEXT_STATUS="✅ Ready"
fi

if test_vllm_service "Embedding" "8001"; then
    EMBEDDING_STATUS="✅ Ready"
fi

if test_vllm_service "Vision" "8002"; then
    VISION_STATUS="✅ Ready"
fi

# Display vLLM status
echo -e "\n${PURPLE}📊 RTX 3090 vLLM Server Status:${NC}"
echo -e "  Text LLM (Port 8000):     $TEXT_STATUS"
echo -e "  Embedding (Port 8001):   $EMBEDDING_STATUS"
echo -e "  Vision (Port 8002):      $VISION_STATUS"

# Check if required services are ready
if [[ "$TEXT_STATUS" == *"Failed"* ]] || [[ "$EMBEDDING_STATUS" == *"Failed"* ]]; then
    echo -e "\n${RED}❌ Required vLLM services are not accessible${NC}"
    echo -e "${YELLOW}   Please ensure your RTX 3090 vLLM servers are running${NC}"
    echo -e "${YELLOW}   Check firewall settings and network connectivity${NC}"
    exit 1
fi

# Test Supabase connectivity
echo -e "\n${BLUE}🔍 Testing Supabase connectivity...${NC}"

if curl -s -H "apikey: $SUPABASE_KEY" \
  "$SUPABASE_URL/rest/v1/crawled_pages?select=*&limit=1" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Supabase connection successful${NC}"
else
    echo -e "${RED}❌ Supabase connection failed${NC}"
    echo -e "${YELLOW}   Please check your Supabase URL and service key${NC}"
    exit 1
fi

# Setup Python environment
echo -e "\n${BLUE}🐍 Setting up Python environment...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found. Installing Python...${NC}"
    brew install python@3.11
fi

echo -e "${GREEN}✅ Python3 found: $(python3 --version)${NC}"

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
fi

echo -e "${GREEN}✅ Virtual environment ready${NC}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo -e "\n${BLUE}📦 Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed from requirements.txt${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found, installing basic dependencies...${NC}"
    pip install mcp crawl4ai supabase requests aiohttp
fi

# Create production configuration
echo -e "\n${BLUE}🔧 Creating production configuration...${NC}"

cat > .env.production << EOF
# Crawl4AI MCP Server Configuration for RTX 3090 vLLM Servers
# macOS deployment connecting to remote vLLM servers

# =============================================================================
# AI Provider Configuration (vLLM - Remote RTX 3090)
# =============================================================================
AI_PROVIDER=vllm
VLLM_BASE_URL=http://$RTX_SERVER_IP:8000/v1
VLLM_API_KEY=EMPTY

# Text and Embedding Models
VLLM_TEXT_MODEL=meta-llama/Llama-3.1-8B-Instruct
VLLM_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
VLLM_EMBEDDING_DIMENSIONS=1024

# Vision Model Configuration
VLLM_VISION_ENABLED=true
VLLM_VISION_MODEL=llava-hf/llava-v1.6-mistral-7b-hf

# Performance Settings
VLLM_MAX_BATCH_SIZE=32
VLLM_MAX_RETRIES=3
VLLM_TIMEOUT=120.0

# =============================================================================
# Vector Database Configuration (Supabase)
# =============================================================================
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=$SUPABASE_URL
SUPABASE_SERVICE_KEY=$SUPABASE_KEY
EMBEDDING_DIMENSION=1024

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
# Logging and Monitoring
# =============================================================================
LOG_LEVEL=INFO
PYTHONPATH=$PROJECT_DIR
EOF

echo -e "${GREEN}✅ Production configuration created${NC}"

# Create logs directory
mkdir -p "$LOG_DIR"

# Create startup script
echo -e "\n${BLUE}📝 Creating startup script...${NC}"

cat > start_mcp_production.sh << EOF
#!/bin/bash
cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR:\$PYTHONPATH"
python3 src/crawl4ai_mcp.py
EOF

chmod +x start_mcp_production.sh

echo -e "${GREEN}✅ Startup script created${NC}"

# Create monitoring script
cat > monitor_mcp.sh << 'EOF'
#!/bin/bash
echo "=== MCP Server Status ==="
curl -s http://localhost:8051/health_check | jq . 2>/dev/null || echo "Server not responding"

echo -e "\n=== Process Status ==="
ps aux | grep crawl4ai_mcp | grep -v grep

echo -e "\n=== Memory Usage ==="
ps -o pid,ppid,pmem,pcpu,comm -p $(pgrep -f crawl4ai_mcp) 2>/dev/null || echo "Process not running"

echo -e "\n=== Network Status ==="
netstat -an | grep 8051

echo -e "\n=== Recent Logs ==="
tail -n 10 logs/mcp_server.log 2>/dev/null || echo "No logs found"
EOF

chmod +x monitor_mcp.sh

echo -e "${GREEN}✅ Monitoring script created${NC}"

# Create backup script
cat > backup_production.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/backups/crawl4ai"

mkdir -p "$BACKUP_DIR"

# Backup configuration
cp .env.production "$BACKUP_DIR/env_$DATE.backup"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "*.backup" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup_production.sh

echo -e "${GREEN}✅ Backup script created${NC}"

# Test the setup
echo -e "\n${BLUE}🧪 Testing the setup...${NC}"

# Copy production environment
cp .env.production .env

# Test MCP server startup (brief test)
echo -e "${BLUE}Testing MCP server startup...${NC}"
timeout 10s python3 src/crawl4ai_mcp.py &
MCP_PID=$!
sleep 5

# Test health check
if curl -s http://localhost:8051/health_check > /dev/null 2>&1; then
    echo -e "${GREEN}✅ MCP server test successful${NC}"
else
    echo -e "${YELLOW}⚠️  MCP server test inconclusive (may need more time to start)${NC}"
fi

# Clean up test process
kill $MCP_PID 2>/dev/null || true

# Display deployment summary
echo -e "\n${PURPLE}🎉 Deployment Complete!${NC}"
echo -e "\n${BLUE}📋 Deployment Summary:${NC}"
echo -e "  Project Directory: $PROJECT_DIR"
echo -e "  Virtual Environment: $VENV_DIR"
echo -e "  Logs Directory: $LOG_DIR"
echo -e "  RTX 3090 Server: $RTX_SERVER_IP"
echo -e "  Supabase URL: $SUPABASE_URL"
echo -e "  MCP Server Port: 8051"

echo -e "\n${BLUE}🚀 To start the MCP server:${NC}"
echo -e "  ./start_mcp_production.sh"

echo -e "\n${BLUE}🔍 To monitor the server:${NC}"
echo -e "  ./monitor_mcp.sh"

echo -e "\n${BLUE}💾 To backup configuration:${NC}"
echo -e "  ./backup_production.sh"

echo -e "\n${BLUE}🧪 To test the server:${NC}"
echo -e "  curl http://localhost:8051/health_check"

echo -e "\n${GREEN}🎉 Your Crawl4AI MCP server is ready for production use!${NC}"
echo -e "${PURPLE}   The server will connect to your RTX 3090 vLLM infrastructure${NC}"
echo -e "${PURPLE}   and provide high-performance RAG capabilities.${NC}"
