# Crawl4AI macOS Deployment with RTX 3090 vLLM Servers

This guide shows you how to deploy Crawl4AI MCP server on macOS without Docker, connecting to your remote RTX 3090 vLLM servers.

## 🎯 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │    │  Crawl4AI MCP   │    │   RTX 3090      │
│   (Claude, etc) │◄──►│   Server (macOS)│◄──►│   vLLM Servers  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │ Vector Database │
                       │ (Supabase/SQLite)│
                       └─────────────────┘
```

## 🚀 Quick Deployment

### 1. Prerequisites

```bash
# Check macOS version
sw_vers

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.8+
brew install python@3.11

# Install Git
brew install git
```

### 2. Clone and Setup Project

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd mcp-crawl4ai-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure for RTX 3090 vLLM Servers

Create the configuration file:

```bash
# Create production configuration
cat > .env.production << 'EOF'
# Crawl4AI MCP Server Configuration for RTX 3090 vLLM Servers
# macOS deployment connecting to remote vLLM servers

# =============================================================================
# AI Provider Configuration (vLLM - Remote RTX 3090)
# =============================================================================
AI_PROVIDER=vllm

# vLLM Server Configuration (Update with your RTX 3090 server IPs)
VLLM_BASE_URL=http://your-rtx3090-server:8000/v1
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
# Vector Database Configuration
# =============================================================================
# Option 1: Supabase (Recommended for production)
VECTOR_DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
EMBEDDING_DIMENSION=1024

# Option 2: SQLite (For development)
# VECTOR_DB_PROVIDER=sqlite
# SQLITE_DB_PATH=./production_data.db
# EMBEDDING_DIMENSION=1024

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
PYTHONPATH=/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag

# =============================================================================
# Security (Optional)
# =============================================================================
# Add API key authentication if needed
# MCP_API_KEY=your-secret-api-key
EOF
```

### 4. Update Configuration

Edit the configuration with your specific settings:

```bash
# Edit the configuration file
nano .env.production

# Update these values:
# - VLLM_BASE_URL: Your RTX 3090 server IP
# - SUPABASE_URL: Your Supabase project URL
# - SUPABASE_SERVICE_KEY: Your Supabase service key
```

### 5. Test vLLM Connectivity

```bash
# Test connection to your RTX 3090 vLLM servers
curl http://your-rtx3090-server:8000/v1/models
curl http://your-rtx3090-server:8001/v1/models
curl http://your-rtx3090-server:8002/v1/models

# Test embedding
curl -X POST http://your-rtx3090-server:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-large-en-v1.5", "input": "test"}'

# Test text generation
curl -X POST http://your-rtx3090-server:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 10}'
```

### 6. Start MCP Server

```bash
# Copy production environment
cp .env.production .env

# Start MCP server
python3 src/crawl4ai_mcp.py
```

## 🔧 Production Deployment Options

### Option 1: Direct Python Process

```bash
# Create startup script
cat > start_mcp_server.sh << 'EOF'
#!/bin/bash
cd /Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag
source venv/bin/activate
export PYTHONPATH="/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag:$PYTHONPATH"
python3 src/crawl4ai_mcp.py
EOF

chmod +x start_mcp_server.sh

# Start server
./start_mcp_server.sh
```

### Option 2: LaunchAgent (macOS Service)

```bash
# Create LaunchAgent plist
cat > ~/Library/LaunchAgents/com.crawl4ai.mcp.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.crawl4ai.mcp</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/venv/bin/python3</string>
        <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/src/crawl4ai_mcp.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/logs/mcp_server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/logs/mcp_server_error.log</string>
</dict>
</plist>
EOF

# Load the service
launchctl load ~/Library/LaunchAgents/com.crawl4ai.mcp.plist

# Check status
launchctl list | grep crawl4ai
```

### Option 3: PM2 Process Manager

```bash
# Install PM2
npm install -g pm2

# Create PM2 ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'crawl4ai-mcp',
    script: 'src/crawl4ai_mcp.py',
    interpreter: 'python3',
    cwd: '/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag',
    env: {
      PYTHONPATH: '/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag',
      NODE_ENV: 'production'
    },
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }]
};
EOF

# Start with PM2
pm2 start ecosystem.config.js

# Check status
pm2 status
pm2 logs crawl4ai-mcp
```

## 🧪 Testing Your Deployment

### 1. Health Check

```bash
# Test MCP server health
curl http://localhost:8051/health_check

# Expected response:
# {
#   "status": "healthy",
#   "service": "mcp-crawl4ai-rag",
#   "components": {
#     "ai_provider": {"status": "healthy", "provider": "vllm"},
#     "database": {"status": "healthy", "provider": "supabase"}
#   }
# }
```

### 2. Test Crawling

```bash
# Test single page crawl
curl -X POST http://localhost:8051/crawl_single_page \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test smart crawl
curl -X POST http://localhost:8051/smart_crawl_url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "max_depth": 2}'
```

### 3. Test RAG Functionality

```bash
# Test RAG query
curl -X POST http://localhost:8051/perform_rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "match_count": 5}'

# Test code examples search
curl -X POST http://localhost:8051/search_code_examples \
  -H "Content-Type: application/json" \
  -d '{"query": "python function", "match_count": 3}'
```

## 📊 Monitoring and Logging

### 1. Log Management

```bash
# Create log rotation script
cat > rotate_logs.sh << 'EOF'
#!/bin/bash
LOG_DIR="/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/logs"
DATE=$(date +%Y%m%d_%H%M%S)

# Rotate logs
if [ -f "$LOG_DIR/mcp_server.log" ]; then
    mv "$LOG_DIR/mcp_server.log" "$LOG_DIR/mcp_server_$DATE.log"
    gzip "$LOG_DIR/mcp_server_$DATE.log"
fi

# Keep only last 7 days of logs
find "$LOG_DIR" -name "mcp_server_*.log.gz" -mtime +7 -delete
EOF

chmod +x rotate_logs.sh

# Add to crontab for daily rotation
(crontab -l 2>/dev/null; echo "0 2 * * * /Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/rotate_logs.sh") | crontab -
```

### 2. System Monitoring

```bash
# Create monitoring script
cat > monitor_mcp.sh << 'EOF'
#!/bin/bash
echo "=== MCP Server Status ==="
curl -s http://localhost:8051/health_check | jq .

echo -e "\n=== Process Status ==="
ps aux | grep crawl4ai_mcp | grep -v grep

echo -e "\n=== Memory Usage ==="
ps -o pid,ppid,pmem,pcpu,comm -p $(pgrep -f crawl4ai_mcp)

echo -e "\n=== Network Status ==="
netstat -an | grep 8051

echo -e "\n=== Recent Logs ==="
tail -n 10 logs/mcp_server.log
EOF

chmod +x monitor_mcp.sh
```

### 3. Performance Monitoring

```bash
# Install monitoring tools
brew install htop iotop nethogs

# Monitor system resources
htop

# Monitor network
nethogs

# Monitor disk I/O
sudo iotop
```

## 🔒 Security Configuration

### 1. Firewall Setup

```bash
# Enable macOS firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on

# Allow MCP server port
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/venv/bin/python3
```

### 2. API Key Authentication

```bash
# Add to .env.production
MCP_API_KEY=your-secret-api-key-here

# Test with API key
curl -H "Authorization: Bearer your-secret-api-key-here" \
  http://localhost:8051/health_check
```

### 3. SSL/TLS Configuration

```bash
# Install nginx for SSL termination
brew install nginx

# Create nginx configuration
cat > /usr/local/etc/nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream mcp_backend {
        server 127.0.0.1:8051;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /path/to/your/cert.pem;
        ssl_certificate_key /path/to/your/key.pem;

        location / {
            proxy_pass http://mcp_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

# Start nginx
sudo nginx
```

## 🚀 Production Optimization

### 1. Performance Tuning

```bash
# Optimize Python for production
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Set optimal process limits
ulimit -n 65536
ulimit -u 32768
```

### 2. Database Optimization

```bash
# For SQLite optimization
sqlite3 production_data.db "PRAGMA journal_mode=WAL;"
sqlite3 production_data.db "PRAGMA synchronous=NORMAL;"
sqlite3 production_data.db "PRAGMA cache_size=10000;"
sqlite3 production_data.db "PRAGMA temp_store=MEMORY;"
```

### 3. Network Optimization

```bash
# Optimize network settings
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.core.netdev_max_backlog=5000
```

## 🔄 Backup and Recovery

### 1. Database Backup

```bash
# Create backup script
cat > backup_database.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/Users/radoslavbali-jencik/backups/crawl4ai"

mkdir -p "$BACKUP_DIR"

# Backup SQLite database
if [ -f "production_data.db" ]; then
    cp production_data.db "$BACKUP_DIR/production_data_$DATE.db"
    gzip "$BACKUP_DIR/production_data_$DATE.db"
fi

# Backup configuration
cp .env.production "$BACKUP_DIR/env_$DATE.backup"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "*.db.gz" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.backup" -mtime +30 -delete

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup_database.sh

# Schedule daily backups
(crontab -l 2>/dev/null; echo "0 3 * * * /Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag/backup_database.sh") | crontab -
```

### 2. Recovery Procedures

```bash
# Restore from backup
gunzip -c /Users/radoslavbali-jencik/backups/crawl4ai/production_data_20240101_030000.db.gz > production_data.db

# Restore configuration
cp /Users/radoslavbali-jencik/backups/crawl4ai/env_20240101_030000.backup .env.production
```

## 🎉 Success Indicators

Your deployment is working correctly when:

1. ✅ MCP server starts without errors
2. ✅ Health check returns "healthy" status
3. ✅ Can connect to RTX 3090 vLLM servers
4. ✅ Database operations work correctly
5. ✅ Crawling operations complete successfully
6. ✅ RAG queries return relevant results
7. ✅ System resources are within normal limits

## 📞 Troubleshooting

### Common Issues

#### 1. Connection to RTX 3090 Servers
```bash
# Test network connectivity
ping your-rtx3090-server
telnet your-rtx3090-server 8000
telnet your-rtx3090-server 8001
telnet your-rtx3090-server 8002
```

#### 2. Database Connection Issues
```bash
# Test Supabase connection
curl -H "apikey: your-service-key" \
  "https://your-project.supabase.co/rest/v1/crawled_pages?select=*&limit=1"
```

#### 3. Performance Issues
```bash
# Monitor system resources
top -p $(pgrep -f crawl4ai_mcp)
iostat -x 1
netstat -i
```

Your macOS deployment provides a robust, scalable solution that leverages your powerful RTX 3090 infrastructure while running the MCP server locally for optimal performance and reliability!
