#!/usr/bin/env python3
"""
Test script to search for "file operations" in the crawled content.
"""
import json
import sys
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local', override=True)

# Add src to path
sys.path.insert(0, 'src')

from crawl4ai_mcp_refactored import perform_rag_query, Crawl4AIContext
from database import DatabaseConfig, DatabaseProviderFactory
from crawl4ai import AsyncWebCrawler, BrowserConfig

# Mock the embedding function for search
import utils_abstracted

async def mock_create_embedding(text):
    """Mock embedding function that returns zero embedding"""
    return [0.0] * 1024

# Replace the embedding function
utils_abstracted.create_embedding = mock_create_embedding

# Also need to check search_documents_abstracted function
from utils_abstracted import search_documents_abstracted

class MockContext:
    def __init__(self, user_data):
        self.user_data = user_data

async def test_search_file_operations():
    """Test searching for file operations in crawled content."""
    print("🔍 Testing search for 'file operations'...")
    
    # Initialize components (minimal setup since we just need database)
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    try:
        # Initialize database
        db_config = DatabaseConfig.from_env()
        print(f"📊 Using {db_config.provider.value} database")
        vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
        await vector_db.initialize()
        
        # Create context
        ctx = Crawl4AIContext(
            crawler=crawler, 
            vector_db=vector_db,
            reranking_model=None,
            knowledge_validator=None,
            repo_extractor=None
        )
        
        context = MockContext(ctx)
        
        # Test search
        print("🔎 Searching for 'file operations'...")
        result = await perform_rag_query(
            context, 
            query="file operations",
            match_count=5,
            source_filter=None
        )
        
        # Parse and display result
        result_data = json.loads(result)
        if result_data.get('success'):
            print("✅ Search successful!")
            print(f"📊 Total results: {result_data.get('total_results')}")
            print(f"🔍 Query: {result_data.get('query')}")
            print(f"🔗 Hybrid search: {result_data.get('hybrid_search_used')}")
            
            results = result_data.get('results', [])
            if results:
                print("\n📋 Top results:")
                for i, result in enumerate(results[:3], 1):
                    content = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
                    print(f"\n{i}. Score: {result.get('score', 'N/A')}")
                    print(f"   Source: {result.get('source', 'N/A')}")
                    print(f"   Content: {content}")
            else:
                print("📝 No results found")
        else:
            print("❌ Search failed!")
            print(f"Error: {result_data.get('error')}")
            
        await vector_db.close()
        
    finally:
        await crawler.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(test_search_file_operations())