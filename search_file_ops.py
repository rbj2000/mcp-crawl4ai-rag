#!/usr/bin/env python3
"""
Search for specific file operations content in the crawled Python docs.
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

async def search_file_operations():
    """Search for file operations content."""
    print("🔍 Searching for file operations content...")
    
    try:
        # Initialize database
        db_config = DatabaseConfig.from_env()
        vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
        await vector_db.initialize()
        
        # Search for documents containing file operation keywords
        print("\n🔎 Looking for content with file operation keywords...")
        
        keywords = [
            'open', 'read', 'write', 'file', 'directory', 'path', 
            'mkdir', 'rmdir', 'listdir', 'remove', 'rename',
            'os.open', 'os.close', 'os.read', 'os.write'
        ]
        
        # Get all documents and filter by keywords
        query_embedding = [0.0] * 1024  # Mock embedding
        all_results = await vector_db.search_documents(
            query_embedding=query_embedding,
            match_count=500  # Get all documents
        )
        
        print(f"📊 Searching through {len(all_results)} total documents...")
        
        # Find documents with file operation content
        file_op_docs = []
        for result in all_results:
            content_lower = result.content.lower()
            matching_keywords = [kw for kw in keywords if kw in content_lower]
            if matching_keywords:
                file_op_docs.append({
                    'result': result,
                    'keywords': matching_keywords,
                    'keyword_count': len(matching_keywords)
                })
        
        # Sort by number of matching keywords
        file_op_docs.sort(key=lambda x: x['keyword_count'], reverse=True)
        
        print(f"✅ Found {len(file_op_docs)} documents with file operation content!")
        
        # Show top results
        print(f"\n📋 Top file operation content:")
        for i, doc_info in enumerate(file_op_docs[:8], 1):
            result = doc_info['result']
            keywords = doc_info['keywords']
            
            print(f"\n{i}. Document ID: {result.id}")
            print(f"   Keywords found: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            print(f"   URL: {result.url}")
            
            # Show content snippet highlighting file operations
            content = result.content
            if len(content) > 400:
                # Try to find a good snippet with file operation content
                best_snippet = content[:400]
                for keyword in ['os.', 'file', 'open', 'read', 'write', 'directory']:
                    pos = content.lower().find(keyword)
                    if pos != -1:
                        start = max(0, pos - 100)
                        end = min(len(content), start + 400)
                        best_snippet = content[start:end]
                        break
                content = best_snippet + "..."
            
            print(f"   Content: {content}")
        
        await vector_db.close()
        
    except Exception as e:
        print(f"❌ Error searching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(search_file_operations())