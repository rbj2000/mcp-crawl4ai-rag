#!/usr/bin/env python3
"""
Start the MCP server with proper environment loading.
"""

import os
import sys
import subprocess

def load_env_file(env_file: str = '.env.local'):
    """Load environment variables from file"""
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    return env_vars

def main():
    print("🚀 Starting MCP Crawl4AI RAG Server...")
    
    # Load environment variables
    env_vars = load_env_file('.env.local')
    
    # Set environment variables
    os.environ.update(env_vars)
    
    print(f"📡 AI Provider: {env_vars.get('AI_PROVIDER', 'not set')}")
    print(f"🗄️  Database: {env_vars.get('VECTOR_DB_PROVIDER', 'not set')}")
    print(f"🌐 Server: http://localhost:{env_vars.get('PORT', '8051')}")
    print(f"🔗 SSE Endpoint: http://localhost:{env_vars.get('PORT', '8051')}/sse")
    print(f"❤️  Health Check: http://localhost:{env_vars.get('PORT', '8051')}/health")
    print(f"📊 Status: http://localhost:{env_vars.get('PORT', '8051')}/status")
    print(f"🏓 Ping: http://localhost:{env_vars.get('PORT', '8051')}/ping")
    
    # Add src to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    print("\n" + "="*50)
    print("Starting server... (Press Ctrl+C to stop)")
    print("="*50)
    
    try:
        # Start the server (using refactored version with AI provider abstraction)
        subprocess.run([
            sys.executable, 
            'src/crawl4ai_mcp_refactored.py'
        ], env=os.environ)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")

if __name__ == "__main__":
    main()