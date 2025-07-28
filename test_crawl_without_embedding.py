#!/usr/bin/env python3
"""
Test script to crawl without trying to create embeddings.
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

from crawl4ai_mcp_refactored import crawl_single_page, Crawl4AIContext
from database import DatabaseConfig, DatabaseProviderFactory
from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.extraction_strategy import NoExtractionStrategy

# Mock the embedding function to test crawling without embeddings
import utils_abstracted

async def mock_create_embeddings_batch(texts):
    """Mock embedding function that returns zero embeddings"""
    return [[0.0] * 1024 for _ in texts]

# Replace the embedding function
utils_abstracted.create_embeddings_batch = mock_create_embeddings_batch

class MockContext:
    def __init__(self, user_data):
        self.user_data = user_data

async def test_crawl_without_embeddings():
    """Test crawling without real embeddings."""
    print("🚀 Testing crawl without embeddings...")
    
    # Initialize components
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
        
        # Test crawl
        print("🔍 Crawling https://docs.python.org/3/library/os.html...")
        result = await crawl_single_page(context, 'https://docs.python.org/3/library/os.html')
        
        # Parse and display result
        result_data = json.loads(result)
        if result_data.get('success'):
            print("✅ Crawl successful with mock embeddings!")
            print(f"📄 URL: {result_data.get('url')}")
            print(f"📝 Chunks added: {result_data.get('chunks_added')}")
            print(f"📊 Word count: {result_data.get('word_count')}")
            print(f"🏷️  Source ID: {result_data.get('source_id')}")
            print("🎉 MCP server crawling functionality working!")
        else:
            print("❌ Crawl failed!")
            print(f"Error: {result_data.get('error')}")
            
        await vector_db.close()
        
    finally:
        await crawler.__aexit__(None, None, None)

if __name__ == "__main__":
    asyncio.run(test_crawl_without_embeddings())