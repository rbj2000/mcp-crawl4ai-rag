#!/usr/bin/env python3
"""
Local testing script for MCP Crawl4AI RAG server.
"""

import os
import sys
import asyncio
from typing import Dict, Any

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables
def load_env_file(env_file: str = '.env.local'):
    """Load environment variables from file"""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️  Environment file {env_file} not found")

def validate_configuration():
    """Validate the local configuration"""
    print("\n🔍 Validating configuration...")
    
    # Check database provider
    db_provider = os.getenv('VECTOR_DB_PROVIDER', 'sqlite')
    print(f"   Database Provider: {db_provider}")
    
    # Check AI provider
    ai_provider = os.getenv('AI_PROVIDER', 'openai')
    print(f"   AI Provider: {ai_provider}")
    
    if ai_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY', '')
        if api_key and api_key != 'your_openai_api_key_here':
            print(f"   OpenAI API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
        else:
            print("   ⚠️  OpenAI API Key: Not set or using placeholder")
            return False
    elif ai_provider == 'ollama':
        base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        print(f"   Ollama Base URL: {base_url}")
    
    # Check Neo4j if knowledge graph is enabled
    if os.getenv('USE_KNOWLEDGE_GRAPH', 'false').lower() == 'true':
        neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        print(f"   Neo4j URI: {neo4j_uri}")
        print(f"   Neo4j User: {neo4j_user}")
    
    # Check RAG strategies
    strategies = []
    if os.getenv('USE_CONTEXTUAL_EMBEDDINGS', 'false').lower() == 'true':
        strategies.append('Contextual Embeddings')
    if os.getenv('USE_HYBRID_SEARCH', 'false').lower() == 'true':
        strategies.append('Hybrid Search')
    if os.getenv('USE_AGENTIC_RAG', 'false').lower() == 'true':
        strategies.append('Agentic RAG')
    if os.getenv('USE_RERANKING', 'false').lower() == 'true':
        strategies.append('Reranking')
    if os.getenv('USE_KNOWLEDGE_GRAPH', 'false').lower() == 'true':
        strategies.append('Knowledge Graph')
    
    print(f"   RAG Strategies: {', '.join(strategies) if strategies else 'None'}")
    
    return True

async def test_basic_functionality():
    """Test basic functionality without starting the full server"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test database provider
        db_provider = os.getenv('VECTOR_DB_PROVIDER', 'sqlite')
        if db_provider == 'sqlite':
            from database.sqlite_provider import SQLiteProvider
            provider = SQLiteProvider({'database_path': './test_local.db'})
            await provider.initialize()
            print("   ✅ SQLite provider initialized successfully")
            await provider.close()
        
        # Test AI provider imports
        ai_provider = os.getenv('AI_PROVIDER', 'openai')
        if ai_provider == 'openai':
            from ai_providers.providers.openai_provider import OpenAIProvider
            print("   ✅ OpenAI provider imported successfully")
        elif ai_provider == 'ollama':
            from ai_providers.providers.ollama_provider import OllamaProvider
            print("   ✅ Ollama provider imported successfully")
        
        # Test utils
        if os.path.exists('src/utils_refactored.py'):
            from utils_refactored import validate_ai_provider_config
            config_result = validate_ai_provider_config()
            if config_result.get('valid'):
                print("   ✅ AI provider configuration is valid")
            else:
                print(f"   ⚠️  AI provider configuration issue: {config_result.get('error', 'Unknown')}")
        else:
            print("   ℹ️  Using original utils.py")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def print_usage_guide():
    """Print usage guide for local testing"""
    print("\n" + "="*60)
    print("📚 LOCAL TESTING GUIDE")
    print("="*60)
    
    print("\n1. **Start the MCP Server**:")
    print("   source .venv/bin/activate")
    print("   export $(cat .env.local | grep -v '^#' | xargs)")
    print("   python src/crawl4ai_mcp.py")
    
    print("\n2. **Test with Claude Code**:")
    print("   claude mcp add-json crawl4ai-rag-local '{\"type\":\"http\",\"url\":\"http://localhost:8051/sse\"}' --scope user")
    
    print("\n3. **Test available tools**:")
    print("   - crawl_single_page: Test with any webpage")
    print("   - smart_crawl_url: Test with documentation sites")
    print("   - get_available_sources: See what's indexed")
    print("   - perform_rag_query: Search crawled content")
    
    if os.getenv('USE_KNOWLEDGE_GRAPH', 'false').lower() == 'true':
        print("   - parse_github_repository: Add repos to knowledge graph")
        print("   - check_ai_script_hallucinations: Validate AI code")
        print("   - query_knowledge_graph: Explore knowledge graph")
    
    print("\n4. **Monitor Neo4j** (if using knowledge graph):")
    print("   Open http://localhost:7474 in browser")
    print("   Login: neo4j / testpassword")
    
    print("\n5. **Database Files**:")
    if os.getenv('VECTOR_DB_PROVIDER') == 'sqlite':
        print(f"   SQLite: {os.getenv('SQLITE_DB_PATH', './local_test.db')}")
    
    print("\n6. **Useful test URLs**:")
    print("   - https://docs.python.org/3/library/os.html")
    print("   - https://fastapi.tiangolo.com/")
    print("   - https://github.com/pydantic/pydantic-ai.git (for knowledge graph)")

async def main():
    """Main testing function"""
    print("🚀 MCP Crawl4AI RAG - Local Testing")
    print("="*50)
    
    # Load environment
    load_env_file('.env.local')
    
    # Validate configuration
    if not validate_configuration():
        print("\n❌ Configuration validation failed!")
        print("Please update your .env.local file with proper values.")
        return False
    
    # Test basic functionality
    if not await test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        return False
    
    print("\n✅ All tests passed! Your local setup is ready.")
    
    # Print usage guide
    print_usage_guide()
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)