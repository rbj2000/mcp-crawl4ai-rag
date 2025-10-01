#!/usr/bin/env python3
"""
Browse the crawled content to see what's actually stored.
"""
import sys
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local', override=True)

# Add src to path
sys.path.insert(0, 'src')

from database import DatabaseConfig, DatabaseProviderFactory

async def browse_crawled_content():
    """Browse the content that was crawled and stored."""
    print("📖 Browsing crawled content...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig.from_env()
        print(f"📊 Using {db_config.provider.value} database")
        vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
        await vector_db.initialize()
        
        # Get sources
        sources = await vector_db.get_sources()
        print(f"\n📚 Found {len(sources)} sources:")
        
        for source in sources:
            print(f"\n🏷️  Source: {source.source_id}")
            print(f"   📊 Total words: {source.total_words}")
            print(f"   📝 Summary: {source.summary}")
            print(f"   📅 Created: {source.created_at}")
        
        # Get a sample of documents to see what content looks like
        print(f"\n📄 Sample documents (searching for 'file' related content):")
        
        # Search for documents containing "file" in the content
        query_embedding = [0.0] * 1024  # Mock embedding
        results = await vector_db.search_documents(
            query_embedding=query_embedding,
            match_count=10
        )
        
        file_related = []
        for result in results:
            content_lower = result.content.lower()
            if any(keyword in content_lower for keyword in ['file', 'open', 'read', 'write', 'directory', 'path']):
                file_related.append(result)
        
        print(f"Found {len(file_related)} documents with file-related content:")
        
        for i, result in enumerate(file_related[:5], 1):
            print(f"\n{i}. ID: {result.id}")
            print(f"   URL: {result.url}")
            # Show content snippet
            content = result.content[:300] + "..." if len(result.content) > 300 else result.content
            print(f"   Content: {content}")
            print(f"   Metadata: {result.metadata}")
        
        await vector_db.close()
        
    except Exception as e:
        print(f"❌ Error browsing content: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(browse_crawled_content())