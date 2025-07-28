#!/usr/bin/env python3
"""
Test Ollama connection and embedding generation.
"""

import json
import requests
import sys

def test_ollama_connection():
    """Test connection to Ollama server"""
    base_url = "https://xx85x9b5v0jnxy-11434.proxy.runpod.net"
    
    print("🔍 Testing Ollama connection...")
    
    try:
        # Test server availability
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json()['models']
            print(f"✅ Connected to Ollama server")
            print(f"📋 Available models: {len(models)}")
            for model in models:
                print(f"   - {model['name']}")
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

def test_embedding_generation():
    """Test embedding generation with bge-m3"""
    base_url = "https://xx85x9b5v0jnxy-11434.proxy.runpod.net"
    
    print("\n🧪 Testing embedding generation...")
    
    try:
        # Test embedding generation
        data = {
            "model": "bge-m3",
            "prompt": "Hello, this is a test for embedding generation."
        }
        
        response = requests.post(
            f"{base_url}/api/embeddings", 
            json=data, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('embedding', [])
            print(f"✅ Embedding generated successfully")
            print(f"📏 Embedding dimensions: {len(embedding)}")
            print(f"🔢 First 5 values: {embedding[:5]}")
            return True
        else:
            print(f"❌ Embedding generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False

def test_llm_generation():
    """Test LLM text generation with deepseek-r1-128k"""
    base_url = "https://xx85x9b5v0jnxy-11434.proxy.runpod.net"
    
    print("\n💬 Testing LLM generation...")
    
    try:
        # Test text generation
        data = {
            "model": "deepseek-r1-128k",
            "prompt": "What is machine learning? Please give a brief explanation.",
            "stream": False
        }
        
        response = requests.post(
            f"{base_url}/api/generate", 
            json=data, 
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✅ Text generation successful")
            print(f"📝 Generated text: {generated_text[:200]}...")
            return True
        else:
            print(f"❌ Text generation failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def main():
    print("🦙 Ollama RunPod Connection Test")
    print("="*40)
    
    # Test connection
    if not test_ollama_connection():
        sys.exit(1)
    
    # Test embedding generation
    if not test_embedding_generation():
        print("⚠️  Embedding test failed, but server is accessible")
    
    # Test LLM generation  
    if not test_llm_generation():
        print("⚠️  LLM test failed, but server is accessible")
    
    print("\n🎉 Ollama tests completed!")
    print("\nYour .env.local configuration:")
    print("AI_PROVIDER=ollama")
    print("OLLAMA_BASE_URL=https://xx85x9b5v0jnxy-11434.proxy.runpod.net")
    print("OLLAMA_EMBEDDING_MODEL=bge-m3")
    print("OLLAMA_LLM_MODEL=deepseek-r1-128k")
    print("OLLAMA_EMBEDDING_DIMENSIONS=768  # Based on bge-m3 model")

if __name__ == "__main__":
    main()