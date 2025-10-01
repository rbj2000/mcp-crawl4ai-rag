import json
import time
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from supabase import create_client, Client
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

logger = logging.getLogger(__name__)

class SupabaseProvider(VectorDatabaseProvider):
    """Supabase/pgvector implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[Client] = None
        self.embedding_dimension = config.get("embedding_dimension", 1536)
        logger.info(f"Supabase provider initialized with embedding dimension: {self.embedding_dimension}")
    
    def get_embedding_dimension(self) -> int:
        """Get the configured embedding dimension for this provider"""
        return self.embedding_dimension
    
    async def validate_embedding_dimension(self, dimension: int) -> bool:
        """Validate if the given embedding dimension is compatible"""
        try:
            # Check if we have any existing data
            result = self.client.table("crawled_pages").select("embedding").limit(1).execute()
            
            if not result.data:
                return True  # No existing data
            
            # Check the actual dimension of stored embeddings
            first_embedding = result.data[0].get("embedding", [])
            if first_embedding and len(first_embedding) != dimension:
                logger.error(
                    f"Embedding dimension mismatch: existing data has {len(first_embedding)} "
                    f"dimensions, requested {dimension}"
                )
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating embedding dimension: {e}")
            return False
    
    async def migrate_dimension(self, new_dimension: int) -> bool:
        """Migrate to a new embedding dimension"""
        logger.warning(
            f"Supabase provider cannot automatically migrate from dimension "
            f"{self.embedding_dimension} to {new_dimension}. "
            f"You need to update the vector column definition and regenerate embeddings."
        )
        
        # Provide migration guidance
        logger.info(
            f"Migration steps:\n"
            f"1. Update vector column: ALTER TABLE crawled_pages ALTER COLUMN embedding TYPE vector({new_dimension});\n"
            f"2. Update vector column: ALTER TABLE code_examples ALTER COLUMN embedding TYPE vector({new_dimension});\n"
            f"3. Regenerate and re-insert all embeddings with the new dimension\n"
            f"4. Update vector indexes if needed"
        )
        
        return False

    async def initialize(self) -> None:
        """Initialize the Supabase client"""
        self.client = create_client(
            self.config["url"],
            self.config["service_key"]
        )
        
        # Test connection
        await self.health_check()
        
        # Validate existing data compatibility
        if not await self.validate_embedding_dimension(self.embedding_dimension):
            raise ValueError(
                f"Existing data has incompatible embedding dimension. "
                f"Expected {self.embedding_dimension}. "
                f"Use migrate_dimension method for migration guidance."
            )
    
    async def close(self) -> None:
        """Close the database connection"""
        # Supabase client doesn't need explicit closing
        self.client = None
    
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive"""
        try:
            # Simple query to test connection
            result = self.client.table("crawled_pages").select("count", count="exact").limit(0).execute()
            return True
        except Exception as e:
            print(f"Supabase health check failed: {e}")
            return False
    
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        batch_size: int = 20
    ) -> None:
        """Add document chunks to Supabase"""
        if not documents:
            return
        
        # Validate embedding dimensions
        for doc in documents:
            if len(doc.embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Document embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(doc.embedding)} for URL {doc.url}"
                )
        
        # Get unique URLs to delete existing records
        unique_urls = list(set(doc.url for doc in documents))
        
        # Delete existing records
        try:
            if unique_urls:
                self.client.table("crawled_pages").delete().in_("url", unique_urls).execute()
        except Exception as e:
            print(f"Batch delete failed: {e}")
            # Fallback: delete one by one
            for url in unique_urls:
                try:
                    self.client.table("crawled_pages").delete().eq("url", url).execute()
                except Exception as inner_e:
                    print(f"Error deleting record for URL {url}: {inner_e}")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_documents = documents[i:batch_end]
            
            batch_data = []
            for doc in batch_documents:
                data = {
                    "url": doc.url,
                    "chunk_number": doc.chunk_number,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "source_id": doc.source_id,
                    "embedding": doc.embedding
                }
                batch_data.append(data)
            
            # Insert with retry logic
            await self._insert_with_retry(
                table="crawled_pages", 
                data=batch_data, 
                max_retries=3
            )
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = self.client.rpc('match_crawled_pages', params).execute()
            
            return [
                VectorSearchResult(
                    id=str(row.get('id', '')),
                    content=row.get('content', ''),
                    metadata=row.get('metadata', {}),
                    similarity=row.get('similarity', 0.0),
                    url=row.get('url', ''),
                    chunk_number=row.get('chunk_number', 0),
                    source_id=row.get('source_id', '')
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def add_code_examples(
        self, 
        examples: List[CodeExample], 
        batch_size: int = 20
    ) -> None:
        """Add code examples to Supabase"""
        if not examples:
            return
        
        # Validate embedding dimensions
        for ex in examples:
            if len(ex.embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Code example embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(ex.embedding)} for URL {ex.url}"
                )
        
        # Delete existing records for these URLs
        unique_urls = list(set(ex.url for ex in examples))
        for url in unique_urls:
            try:
                self.client.table('code_examples').delete().eq('url', url).execute()
            except Exception as e:
                print(f"Error deleting existing code examples for {url}: {e}")
        
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch_end = min(i + batch_size, len(examples))
            batch_examples = examples[i:batch_end]
            
            batch_data = []
            for ex in batch_examples:
                data = {
                    'url': ex.url,
                    'chunk_number': ex.chunk_number,
                    'content': ex.content,
                    'summary': ex.summary,
                    'metadata': ex.metadata,
                    'source_id': ex.source_id,
                    'embedding': ex.embedding
                }
                batch_data.append(data)
            
            await self._insert_with_retry(
                table="code_examples", 
                data=batch_data, 
                max_retries=3
            )
    
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar code examples"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        try:
            params = {
                'query_embedding': query_embedding,
                'match_count': match_count
            }
            
            if filter_metadata:
                params['filter'] = filter_metadata
            
            result = self.client.rpc('match_code_examples', params).execute()
            
            return [
                VectorSearchResult(
                    id=str(row.get('id', '')),
                    content=row.get('content', ''),
                    metadata=row.get('metadata', {}),
                    similarity=row.get('similarity', 0.0),
                    url=row.get('url', ''),
                    chunk_number=row.get('chunk_number', 0),
                    source_id=row.get('source_id', '')
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error searching code examples: {e}")
            return []
    
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        try:
            result = self.client.from_('sources').select('*').order('source_id').execute()
            
            return [
                SourceInfo(
                    source_id=row.get("source_id"),
                    summary=row.get("summary"),
                    total_words=row.get("total_words", 0),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                for row in (result.data or [])
            ]
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    async def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int
    ) -> None:
        """Update source information"""
        try:
            # Try to update existing source
            result = self.client.table('sources').update({
                'summary': summary,
                'total_words': word_count,
                'updated_at': 'now()'
            }).eq('source_id', source_id).execute()
            
            # If no rows were updated, insert new source
            if not result.data:
                self.client.table('sources').insert({
                    'source_id': source_id,
                    'summary': summary,
                    'total_words': word_count
                }).execute()
                
        except Exception as e:
            print(f"Error updating source {source_id}: {e}")
    
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        try:
            # Delete from both tables
            self.client.table("crawled_pages").delete().in_("url", urls).execute()
            self.client.table("code_examples").delete().in_("url", urls).execute()
        except Exception as e:
            print(f"Error deleting by URLs: {e}")
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        # Get vector search results
        vector_results = await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        keyword_query_builder = self.client.from_('crawled_pages')\
            .select('id, url, chunk_number, content, metadata, source_id')\
            .ilike('content', f'%{keyword_query}%')
        
        # Apply source filter if provided
        if filter_metadata and 'source' in filter_metadata:
            keyword_query_builder = keyword_query_builder.eq('source_id', filter_metadata['source'])
        
        keyword_response = keyword_query_builder.limit(match_count * 2).execute()
        keyword_results = keyword_response.data if keyword_response.data else []
        
        # Combine results
        return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    async def hybrid_search_code_examples(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Perform hybrid search for code examples"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        # Get vector search results
        vector_results = await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        keyword_query_builder = self.client.from_('code_examples')\
            .select('id, url, chunk_number, content, summary, metadata, source_id')\
            .or_(f'content.ilike.%{keyword_query}%,summary.ilike.%{keyword_query}%')
        
        # Apply source filter if provided
        if filter_metadata and 'source' in filter_metadata:
            keyword_query_builder = keyword_query_builder.eq('source_id', filter_metadata['source'])
        
        keyword_response = keyword_query_builder.limit(match_count * 2).execute()
        keyword_results = keyword_response.data if keyword_response.data else []
        
        # Combine results
        return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[VectorSearchResult], 
        keyword_results: List[Dict], 
        match_count: int
    ) -> List[VectorSearchResult]:
        """Combine vector and keyword search results"""
        seen_ids = set()
        combined_results = []
        
        # Convert keyword results to VectorSearchResult format
        keyword_search_results = []
        for kr in keyword_results:
            keyword_search_results.append(VectorSearchResult(
                id=str(kr['id']),
                url=kr['url'],
                chunk_number=kr['chunk_number'],
                content=kr.get('content', ''),
                metadata=kr.get('metadata', {}),
                source_id=kr.get('source_id', ''),
                similarity=0.5  # Default similarity for keyword-only matches
            ))
        
        # Add items that appear in both searches first (boost their scores)
        vector_ids = {r.id for r in vector_results}
        for kr in keyword_search_results:
            if kr.id in vector_ids and kr.id not in seen_ids:
                # Find the vector result to get similarity score
                for vr in vector_results:
                    if vr.id == kr.id:
                        # Boost similarity score for items in both results
                        vr.similarity = min(1.0, vr.similarity * 1.2)
                        combined_results.append(vr)
                        seen_ids.add(kr.id)
                        break
        
        # Add remaining vector results
        for vr in vector_results:
            if vr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(vr)
                seen_ids.add(vr.id)
        
        # Add pure keyword matches if needed
        for kr in keyword_search_results:
            if kr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(kr)
                seen_ids.add(kr.id)
        
        return combined_results[:match_count]
    
    async def _insert_with_retry(
        self, 
        table: str, 
        data: List[Dict[str, Any]], 
        max_retries: int = 3
    ) -> None:
        """Insert data with retry logic"""
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                self.client.table(table).insert(data).execute()
                return
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Try inserting records individually
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in data:
                        try:
                            self.client.table(table).insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(f"Failed to insert individual record: {individual_error}")
                    
                    if successful_inserts > 0:
                        print(f"Successfully inserted {successful_inserts}/{len(data)} records individually")