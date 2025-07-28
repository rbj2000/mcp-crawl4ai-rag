#!/usr/bin/env python3
"""
Search for specific information about os.walk() in the crawled content.
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

async def search_os_walk():
    """Search for os.walk() specific content."""
    print("🔍 Searching for os.walk() information...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig.from_env()
        vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
        await vector_db.initialize()
        
        # Search for documents containing os.walk
        query_embedding = [0.0] * 1024  # Mock embedding
        all_results = await vector_db.search_documents(
            query_embedding=query_embedding,
            match_count=500  # Get all documents
        )
        
        # Find documents specifically about os.walk
        walk_docs = []
        for result in all_results:
            content = result.content
            content_lower = content.lower()
            
            # Look for os.walk mentions
            if 'os.walk' in content_lower or 'walk(' in content_lower:
                walk_docs.append(result)
        
        print(f"✅ Found {len(walk_docs)} documents mentioning os.walk()")
        
        if walk_docs:
            print(f"\n📖 **What does os.walk() do?**\n")
            
            # Show the most relevant content
            for i, result in enumerate(walk_docs[:3], 1):
                content = result.content
                print(f"**Result {i}:**")
                print(f"{content}\n")
                print("-" * 80 + "\n")
        else:
            print("❌ No specific os.walk() documentation found")
        
        await vector_db.close()
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(search_os_walk())