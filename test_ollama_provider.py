#!/usr/bin/env python3
"""Test script for the Ollama AI provider implementation"""

import asyncio
import logging
import sys
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, 'src')

from ai_providers.base import AIProvider
from ai_providers.factory import AIProviderFactory


async def test_ollama_provider():
    """Test the Ollama provider implementation"""
    
    # Configuration for Ollama
    config = {
        "base_url": "http://localhost:11434",
        "embedding_model": "mxbai-embed-large",
        "llm_model": "llama3.1",
        "timeout": 120.0,
        "auto_pull_models": True,
        "batch_size": 5
    }
    
    print("🦙 Testing Ollama AI Provider Implementation")
    print("=" * 50)
    
    try:
        # Check if Ollama provider is available
        if not AIProviderFactory.is_provider_available(AIProvider.OLLAMA):
            print("❌ Ollama provider not registered")
            return False
        
        print("✅ Ollama provider is registered")
        
        # Create hybrid provider
        print("\n📦 Creating Ollama hybrid provider...")
        provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, config)
        print(f"✅ Created provider: {type(provider).__name__}")
        
        # Initialize provider
        print("\n🚀 Initializing provider...")
        await provider.initialize()
        print("✅ Provider initialized successfully")
        
        # Health check
        print("\n❤️  Performing health check...")
        health = await provider.health_check()
        print(f"Health Status: {'✅ Healthy' if health.is_healthy else '❌ Unhealthy'}")
        print(f"Response Time: {health.response_time_ms:.1f}ms")
        if health.error_message:
            print(f"Error: {health.error_message}")
        if health.metadata:
            print(f"Metadata: {health.metadata}")
        
        if not health.is_healthy:
            print("❌ Health check failed - cannot continue tests")
            await provider.close()
            return False
        
        # Test embedding functionality
        print("\n🔍 Testing embedding functionality...")
        test_texts = [
            "Hello, world!",
            "This is a test of the Ollama embedding provider.",
            "Python is a great programming language for AI applications.",
            "",  # Test empty text handling
            "A" * 1000  # Test long text
        ]
        
        # Single embedding
        print("Testing single embedding...")
        embedding_result = await provider.create_embedding(test_texts[0])
        print(f"✅ Single embedding: {embedding_result.dimensions} dimensions")
        print(f"Model: {embedding_result.model}")
        print(f"First 5 values: {embedding_result.embedding[:5]}")
        
        # Batch embeddings
        print("Testing batch embeddings...")
        batch_results = await provider.create_embeddings_batch(test_texts)
        print(f"✅ Batch embeddings: {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"  Text {i}: {result.dimensions} dims, model: {result.model}")
        
        # Test LLM functionality
        print("\n🤖 Testing LLM functionality...")
        
        # Simple generation
        messages = [
            {"role": "user", "content": "Hello! Please respond with exactly 'Hello there!' and nothing else."}
        ]
        
        print("Testing text generation...")
        llm_result = await provider.generate(messages, max_tokens=10)
        print(f"✅ Generated text: '{llm_result.content}'")
        print(f"Model: {llm_result.model}")
        print(f"Input tokens: {llm_result.input_tokens}")
        print(f"Output tokens: {llm_result.output_tokens}")
        print(f"Finish reason: {llm_result.finish_reason}")
        
        # Streaming generation
        print("Testing streaming generation...")
        stream_messages = [
            {"role": "user", "content": "Count from 1 to 5, each number on a new line."}
        ]
        
        accumulated_content = ""
        chunk_count = 0
        async for chunk in provider.generate_stream(stream_messages, max_tokens=50):
            chunk_count += 1
            if chunk.content:  # Only print non-empty chunks
                accumulated_content += chunk.content
                print(f"Chunk {chunk_count}: '{chunk.content}'")
            
            if chunk.metadata and chunk.metadata.get("is_final"):
                print(f"✅ Streaming complete: {chunk_count} chunks")
                print(f"Final content: '{accumulated_content}'")
                break
        
        # Test model information
        print("\n📊 Testing model information...")
        
        # Available models
        try:
            available_models = await provider.list_available_models()
            print("✅ Available models:")
            print(f"  Embedding models: {available_models.get('embedding_models', [])}")
            print(f"  LLM models: {available_models.get('llm_models', [])}")
            print(f"  Total models: {len(available_models.get('all_models', []))}")
        except Exception as e:
            print(f"⚠️  Could not list models: {e}")
        
        # Provider metrics
        print("\n📈 Testing provider metrics...")
        metrics = provider.get_metrics()
        print("✅ Provider metrics:")
        print(f"  Embedding requests: {metrics['embedding']['request_count']}")
        print(f"  LLM requests: {metrics['llm']['request_count']}")
        print(f"  Total errors: {metrics['combined']['total_errors']}")
        print(f"  Models pulled: {metrics['combined']['total_model_pulls']}")
        
        # Test error handling
        print("\n🛡️  Testing error handling...")
        
        # Test with invalid model
        try:
            await provider.create_embedding("test", model="nonexistent-model")
            print("⚠️  Expected error for invalid model, but didn't get one")
        except Exception as e:
            print(f"✅ Correctly handled invalid model: {type(e).__name__}")
        
        # Test with empty text
        try:
            result = await provider.create_embedding("")
            print(f"✅ Empty text handled: {len(result.embedding)} dims (should be zeros)")
        except Exception as e:
            print(f"✅ Empty text error handled: {type(e).__name__}")
        
        print("\n🧹 Cleaning up...")
        await provider.close()
        print("✅ Provider closed successfully")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


async def test_provider_factory():
    """Test the provider factory integration"""
    print("\n🏭 Testing Provider Factory Integration")
    print("=" * 50)
    
    # List all available providers
    all_providers = AIProviderFactory.list_all_providers()
    print("Available providers:")
    for provider_type, providers in all_providers.items():
        print(f"  {provider_type}: {providers}")
    
    # Check Ollama capabilities
    capabilities = AIProviderFactory.get_provider_capabilities(AIProvider.OLLAMA)
    print(f"\nOllama capabilities: {capabilities}")
    
    # Test creating different provider types
    config = {"base_url": "http://localhost:11434"}
    
    try:
        embedding_provider = AIProviderFactory.create_embedding_provider(AIProvider.OLLAMA, config)
        print(f"✅ Created embedding provider: {type(embedding_provider).__name__}")
        
        llm_provider = AIProviderFactory.create_llm_provider(AIProvider.OLLAMA, config)
        print(f"✅ Created LLM provider: {type(llm_provider).__name__}")
        
        hybrid_provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, config)
        print(f"✅ Created hybrid provider: {type(hybrid_provider).__name__}")
        
        # Test auto provider selection
        auto_provider = AIProviderFactory.create_provider(AIProvider.OLLAMA, config)
        print(f"✅ Auto-created provider: {type(auto_provider).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Ollama AI Provider Test Suite")
    print("=" * 60)
    
    # Check if Ollama service is likely running
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                if response.status == 200:
                    print("✅ Ollama service appears to be running")
                else:
                    print(f"⚠️  Ollama service responded with status {response.status}")
    except Exception as e:
        print(f"❌ Could not connect to Ollama service: {e}")
        print("Please ensure Ollama is installed and running on localhost:11434")
        print("Install: https://ollama.ai/")
        print("Run: ollama serve")
        return False
    
    # Run tests
    success = True
    
    # Test factory integration
    success &= await test_provider_factory()
    
    # Test full provider functionality
    success &= await test_ollama_provider()
    
    if success:
        print("\n🎉 All tests passed! Ollama provider is ready to use.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)