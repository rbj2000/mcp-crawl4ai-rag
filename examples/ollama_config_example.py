#!/usr/bin/env python3
"""Example configuration and usage of the Ollama AI provider"""

import asyncio
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example configurations for different use cases

# Basic configuration - good for getting started
BASIC_CONFIG = {
    "base_url": "http://localhost:11434",
    "embedding_model": "mxbai-embed-large",  # Good balance of speed/quality
    "llm_model": "llama3.1",                 # Latest Llama model
    "auto_pull_models": True,                # Download models automatically
    "timeout": 120.0,                        # 2 minute timeout
    "batch_size": 5                          # Small batch for testing
}

# Performance-optimized configuration
PERFORMANCE_CONFIG = {
    "base_url": "http://localhost:11434",
    "embedding_model": "all-minilm",         # Fastest embedding model
    "llm_model": "phi3:mini",                # Lightweight LLM
    "auto_pull_models": True,
    "timeout": 60.0,                         # Shorter timeout
    "batch_size": 20,                        # Larger batches
    "temperature": 0.3,                      # More deterministic
    "max_tokens": 500                        # Shorter responses
}

# Quality-focused configuration
QUALITY_CONFIG = {
    "base_url": "http://localhost:11434", 
    "embedding_model": "bge-large",          # High-quality embeddings
    "llm_model": "llama3.1:70b",             # Large, high-quality model
    "auto_pull_models": True,
    "timeout": 600.0,                        # Long timeout for large model
    "batch_size": 3,                         # Smaller batches for large model
    "temperature": 0.7,                      # Balanced creativity
    "max_tokens": 2000                       # Longer responses
}

# Code-focused configuration
CODE_CONFIG = {
    "base_url": "http://localhost:11434",
    "embedding_model": "mxbai-embed-large",  # Good for code embeddings
    "llm_model": "codellama:13b",            # Specialized code model
    "auto_pull_models": True,
    "timeout": 300.0,
    "batch_size": 5,
    "temperature": 0.1,                      # Very deterministic for code
    "max_tokens": 1500
}

# Resource-constrained configuration (low memory)
LIGHTWEIGHT_CONFIG = {
    "base_url": "http://localhost:11434",
    "embedding_model": "all-minilm",         # Small embedding model (46MB)
    "llm_model": "phi3:mini",                # Small LLM (2.3GB)
    "auto_pull_models": True,
    "timeout": 90.0,
    "batch_size": 10,
    "temperature": 0.5,
    "max_tokens": 800
}


async def demonstrate_usage():
    """Demonstrate various usage patterns with Ollama provider"""
    
    # Import after adding src to path if needed
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    from ai_providers.factory import AIProviderFactory
    from ai_providers.base import AIProvider
    
    print("🦙 Ollama AI Provider Configuration Examples")
    print("=" * 60)
    
    # Example 1: Basic usage
    print("\n📚 Example 1: Basic Configuration")
    provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, BASIC_CONFIG)
    
    try:
        await provider.initialize()
        
        # Test embedding
        embedding = await provider.create_embedding("Hello, Ollama!")
        print(f"✅ Created embedding: {embedding.dimensions} dimensions")
        
        # Test generation
        messages = [{"role": "user", "content": "Say hello in 3 different languages."}]
        response = await provider.generate(messages, max_tokens=100)
        print(f"✅ Generated response: {response.content[:100]}...")
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ Basic example failed: {e}")
    
    # Example 2: Batch processing
    print("\n📦 Example 2: Batch Processing")
    provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, PERFORMANCE_CONFIG)
    
    try:
        await provider.initialize()
        
        # Batch embedding example
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Ollama enables running language models locally.",
            "Vector databases are useful for similarity search.",
            "RAG combines retrieval with text generation."
        ]
        
        embeddings = await provider.create_embeddings_batch(documents)
        print(f"✅ Created {len(embeddings)} embeddings in batch")
        
        # Show embedding stats
        for i, emb in enumerate(embeddings):
            print(f"  Doc {i}: {emb.dimensions} dims, model: {emb.model}")
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ Batch example failed: {e}")
    
    # Example 3: Streaming generation
    print("\n🌊 Example 3: Streaming Generation")
    provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, BASIC_CONFIG)
    
    try:
        await provider.initialize()
        
        messages = [
            {"role": "user", "content": "Write a short poem about artificial intelligence."}
        ]
        
        print("Streaming response:")
        print("-" * 40)
        
        full_response = ""
        async for chunk in provider.generate_stream(messages, max_tokens=200):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
            
            if chunk.metadata and chunk.metadata.get("is_final"):
                print(f"\n-" * 40)
                print(f"✅ Streaming complete. Total length: {len(full_response)} chars")
                break
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ Streaming example failed: {e}")
    
    # Example 4: Error handling and model management
    print("\n🛠️  Example 4: Model Management")
    provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, BASIC_CONFIG)
    
    try:
        await provider.initialize()
        
        # List available models
        models = await provider.list_available_models()
        print("✅ Available models:")
        print(f"  Embedding: {models.get('embedding_models', [])}")
        print(f"  LLM: {models.get('llm_models', [])}")
        
        # Health check
        health = await provider.health_check()
        print(f"✅ Health check: {'Healthy' if health.is_healthy else 'Unhealthy'}")
        print(f"  Response time: {health.response_time_ms:.1f}ms")
        
        # Provider metrics
        metrics = provider.get_metrics()
        print("✅ Provider metrics:")
        print(f"  Embedding requests: {metrics['embedding']['request_count']}")
        print(f"  LLM requests: {metrics['llm']['request_count']}")
        print(f"  Total errors: {metrics['combined']['total_errors']}")
        
        await provider.close()
        
    except Exception as e:
        print(f"❌ Model management example failed: {e}")


def print_config_recommendations():
    """Print configuration recommendations for different scenarios"""
    
    print("\n🎯 Configuration Recommendations")
    print("=" * 60)
    
    scenarios = [
        ("🏃 Performance-First", PERFORMANCE_CONFIG, "Fast responses, lower quality"),
        ("🎨 Quality-First", QUALITY_CONFIG, "Best quality, slower responses"),  
        ("💻 Code-Focused", CODE_CONFIG, "Optimized for code tasks"),
        ("💾 Resource-Constrained", LIGHTWEIGHT_CONFIG, "Minimal memory usage"),
        ("🚀 Basic/Getting Started", BASIC_CONFIG, "Balanced, good for testing")
    ]
    
    for name, config, description in scenarios:
        print(f"\n{name}")
        print(f"Description: {description}")
        print(f"Embedding Model: {config['embedding_model']}")
        print(f"LLM Model: {config['llm_model']}")
        print(f"Timeout: {config['timeout']}s")
        print(f"Batch Size: {config['batch_size']}")
        
    print("\n📋 Model Selection Guide")
    print("-" * 30)
    print("Embedding Models (by speed):")
    print("  Fast:    all-minilm (384 dims, 46MB)")
    print("  Medium:  nomic-embed-text (768 dims, 274MB)")
    print("  Slow:    mxbai-embed-large (1024 dims, 669MB)")
    print("  Quality: bge-large (1024 dims, 1.34GB)")
    
    print("\nLLM Models (by resource usage):")
    print("  Light:   phi3:mini (2.3GB)")
    print("  Medium:  llama3.1 (4.7GB)")
    print("  Heavy:   llama3.1:70b (40GB)")
    print("  Code:    codellama:13b (7.3GB)")


def show_environment_setup():
    """Show how to set up environment variables"""
    
    print("\n🔧 Environment Setup")
    print("=" * 60)
    
    print("Add these to your .env file:")
    print("""
# Basic Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_LLM_MODEL=llama3.1

# Optional: Advanced Configuration  
OLLAMA_AUTO_PULL_MODELS=true
OLLAMA_TIMEOUT=300
OLLAMA_BATCH_SIZE=10
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=1000

# Optional: Retry Configuration
OLLAMA_MAX_RETRIES=3
OLLAMA_INITIAL_RETRY_DELAY=2.0
OLLAMA_MAX_RETRY_DELAY=60.0
""")

    print("Then use in your code:")
    print("""
import os
from dotenv import load_dotenv

load_dotenv()

config = {
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"),
    "llm_model": os.getenv("OLLAMA_LLM_MODEL", "llama3.1"),
    "auto_pull_models": os.getenv("OLLAMA_AUTO_PULL_MODELS", "true").lower() == "true",
    "timeout": float(os.getenv("OLLAMA_TIMEOUT", "300")),
    "batch_size": int(os.getenv("OLLAMA_BATCH_SIZE", "10")),
    "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "1000"))
}
""")


async def main():
    """Main function to run all examples"""
    
    print("🦙 Ollama AI Provider Examples & Configuration Guide")
    print("=" * 70)
    
    # Check if Ollama is running
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                if response.status == 200:
                    print("✅ Ollama service is running")
                    
                    # Run demonstrations
                    await demonstrate_usage()
                else:
                    print(f"⚠️  Ollama service responded with status {response.status}")
    except Exception as e:
        print(f"❌ Could not connect to Ollama: {e}")
        print("Please install and start Ollama:")
        print("1. Visit https://ollama.ai/ to install")
        print("2. Run 'ollama serve' to start the service")
        print("3. Verify with 'curl http://localhost:11434/api/tags'")
    
    # Show configuration guides regardless of Ollama status
    print_config_recommendations()
    show_environment_setup()
    
    print("\n🎉 Examples complete! Check the documentation for more details.")


if __name__ == "__main__":
    asyncio.run(main())