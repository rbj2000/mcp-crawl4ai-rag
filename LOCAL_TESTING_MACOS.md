# Local Testing Guide for Crawl4AI MCP Server on macOS

This guide shows you how to test the Crawl4AI MCP server locally on macOS without needing the full RTX 3090 vLLM setup.

## 🎯 Local Testing Options

### Option 1: SQLite + Mock AI (Fastest)
- **Database**: SQLite (local file)
- **AI**: Mock responses (no real AI calls)
- **Best for**: Testing crawling and database functionality

### Option 2: SQLite + Ollama (Recommended)
- **Database**: SQLite (local file)
- **AI**: Ollama (local LLM)
- **Best for**: Full RAG testing with local AI

### Option 3: SQLite + OpenAI (Cloud)
- **Database**: SQLite (local file)
- **AI**: OpenAI API (cloud)
- **Best for**: Testing with production-quality AI

## 🚀 Quick Start (Option 1: Mock AI)

### 1. Create Local Environment File

```bash
# Create local testing environment
cat > .env.local << 'EOF'
# Local Testing Configuration for macOS

# =============================================================================
# AI Provider Configuration (Mock for testing)
# =============================================================================
AI_PROVIDER=openai
OPENAI_API_KEY=mock-key-for-testing
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536

# =============================================================================
# Vector Database Configuration (SQLite)
# =============================================================================
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=./local_test.db
EMBEDDING_DIMENSION=1536

# =============================================================================
# RAG Strategy Configuration
# =============================================================================
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# =============================================================================
# Server Configuration
# =============================================================================
HOST=0.0.0.0
PORT=8051
TRANSPORT_MODE=sse

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=DEBUG
PYTHONPATH=/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag
EOF
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional testing dependencies
pip install pytest pytest-asyncio httpx
```

### 3. Start Local Mock Server

```bash
# Create a simple mock server for testing
cat > mock_ai_server.py << 'EOF'
#!/usr/bin/env python3
"""
Mock AI server for local testing
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

class MockAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/v1/embeddings':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Generate mock embedding
            embedding = [random.random() for _ in range(1536)]
            response = {
                "data": [{"embedding": embedding}],
                "model": "text-embedding-3-small"
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/v1/completions':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Generate mock completion
            response = {
                "choices": [{
                    "text": "This is a mock response for testing purposes.",
                    "finish_reason": "stop"
                }],
                "model": "gpt-4o-mini"
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                "data": [{"id": "text-embedding-3-small"}, {"id": "gpt-4o-mini"}]
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), MockAIHandler)
    print("Mock AI server running on http://localhost:8000")
    server.serve_forever()
EOF

# Start mock server in background
python3 mock_ai_server.py &
MOCK_PID=$!
echo "Mock AI server started with PID: $MOCK_PID"
```

### 4. Test the Setup

```bash
# Test mock server
curl http://localhost:8000/v1/models

# Test embedding
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'

# Test completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "gpt-4o-mini", "max_tokens": 10}'
```

### 5. Start MCP Server

```bash
# Copy local environment
cp .env.local .env

# Start MCP server
python3 src/crawl4ai_mcp.py
```

## 🧪 Testing with Ollama (Option 2: Recommended)

### 1. Install Ollama

```bash
# Install Ollama on macOS
brew install ollama

# Or download from: https://ollama.ai/download
```

### 2. Start Ollama and Pull Models

```bash
# Start Ollama service
ollama serve &

# Pull required models
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### 3. Create Ollama Environment

```bash
# Create Ollama environment
cat > .env.ollama << 'EOF'
# Ollama Local Testing Configuration

# =============================================================================
# AI Provider Configuration (Ollama)
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
```

### 4. Test Ollama Setup

```bash
# Test Ollama
curl http://localhost:11434/api/tags

# Test embedding
curl -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nomic-embed-text", "prompt": "test"}'

# Test completion
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "prompt": "Hello", "stream": false}'
```

### 5. Start MCP Server with Ollama

```bash
# Use Ollama environment
cp .env.ollama .env

# Start MCP server
python3 src/crawl4ai_mcp.py
```

## 🧪 Testing with OpenAI (Option 3: Cloud)

### 1. Get OpenAI API Key

```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 2. Create OpenAI Environment

```bash
# Create OpenAI environment
cat > .env.openai << 'EOF'
# OpenAI Cloud Testing Configuration

# =============================================================================
# AI Provider Configuration (OpenAI)
# =============================================================================
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536

# =============================================================================
# Vector Database Configuration (SQLite)
# =============================================================================
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=./local_test.db
EMBEDDING_DIMENSION=1536

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
OPENAI_MAX_BATCH_SIZE=100
OPENAI_TIMEOUT=30.0

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO
PYTHONPATH=/Users/radoslavbali-jencik/IdeaProjects/mcp-crawl4ai-rag
EOF
```

### 3. Test OpenAI Setup

```bash
# Test OpenAI API
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Test embedding
curl -X POST https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'
```

### 4. Start MCP Server with OpenAI

```bash
# Use OpenAI environment
cp .env.openai .env

# Start MCP server
python3 src/crawl4ai_mcp.py
```

## 🔧 Testing Scripts

### 1. Create Test Script

```bash
# Create comprehensive test script
cat > test_local_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Local testing script for Crawl4AI MCP server
"""
import requests
import json
import time
import sys

def test_health_check():
    """Test MCP server health check"""
    try:
        response = requests.get("http://localhost:8051/health_check", timeout=5)
        if response.status_code == 200:
            print("✅ Health check: OK")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_crawl_single_page():
    """Test single page crawling"""
    try:
        data = {"url": "https://httpbin.org/html"}
        response = requests.post("http://localhost:8051/crawl_single_page", 
                               json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Single page crawl: OK")
                return True
            else:
                print(f"❌ Single page crawl failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Single page crawl failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Single page crawl failed: {e}")
        return False

def test_rag_query():
    """Test RAG query"""
    try:
        data = {"query": "test query", "match_count": 3}
        response = requests.post("http://localhost:8051/perform_rag_query", 
                               json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ RAG query: OK")
                return True
            else:
                print(f"❌ RAG query failed: {result.get('error')}")
                return False
        else:
            print(f"❌ RAG query failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ RAG query failed: {e}")
        return False

def test_get_sources():
    """Test getting available sources"""
    try:
        response = requests.post("http://localhost:8051/get_available_sources", 
                               timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✅ Get sources: OK")
                return True
            else:
                print(f"❌ Get sources failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Get sources failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get sources failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Crawl4AI MCP Server...")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Get Sources", test_get_sources),
        ("Single Page Crawl", test_crawl_single_page),
        ("RAG Query", test_rag_query),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_local_setup.py
```

### 2. Run Tests

```bash
# Run the test script
python3 test_local_setup.py
```

## 🔍 Manual Testing

### 1. Test MCP Server Directly

```bash
# Test health check
curl http://localhost:8051/health_check

# Test crawling
curl -X POST http://localhost:8051/crawl_single_page \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Test RAG query
curl -X POST http://localhost:8051/perform_rag_query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "match_count": 5}'
```

### 2. Test with MCP Client

```bash
# Install MCP client
pip install mcp

# Test with MCP client
mcp-client http://localhost:8051
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using port 8051
lsof -i :8051

# Kill the process
kill -9 $(lsof -t -i:8051)

# Or use a different port
export PORT=8052
```

#### 2. Database Issues
```bash
# Check SQLite database
sqlite3 local_test.db ".tables"

# Reset database
rm local_test.db
```

#### 3. Python Dependencies
```bash
# Check installed packages
pip list | grep -E "(mcp|crawl4ai|supabase)"

# Install missing packages
pip install -r requirements.txt
```

#### 4. Ollama Issues
```bash
# Check Ollama status
ollama list

# Restart Ollama
pkill ollama
ollama serve &
```

## 📊 Performance Expectations

### Local Testing Performance
- **Crawling**: 1-5 pages/second
- **Embeddings**: 10-50 texts/second
- **RAG Queries**: 1-3 seconds
- **Memory Usage**: 1-2GB RAM

### Resource Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 1GB+ for models and database
- **Network**: Internet connection for crawling

## 🎉 Success Indicators

Your local setup is working when:

1. ✅ MCP server starts without errors
2. ✅ Health check returns "healthy"
3. ✅ Can crawl web pages successfully
4. ✅ Can perform RAG queries
5. ✅ Database operations work correctly

## 📝 Next Steps

Once local testing works:

1. **Test with real data**: Crawl actual websites
2. **Test RAG functionality**: Query crawled content
3. **Test different AI providers**: Try Ollama vs OpenAI
4. **Scale up**: Move to your RTX 3090 setup
5. **Production**: Deploy with Supabase and vLLM

Your local macOS setup provides a perfect testing environment before deploying to your powerful RTX 3090 infrastructure!
