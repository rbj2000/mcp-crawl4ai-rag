import asyncio
from typing import List, Dict, Any, Optional
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

class PineconeProvider(VectorDatabaseProvider):
    """Pinecone vector database implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index = None
        self.code_index = None
        self.documents_index_name = config["index_name"]
        self.code_examples_index_name = f"{config['index_name']}-code"
        
        # Store sources in memory since Pinecone doesn't support traditional tables
        self.sources_cache = {}
    
    async def initialize(self) -> None:
        """Initialize Pinecone connection"""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.config["api_key"],
                environment=self.config["environment"]
            )
            
            # Create indexes if they don't exist
            await self._ensure_indexes_exist()
            
            self.index = pinecone.Index(self.documents_index_name)
            self.code_index = pinecone.Index(self.code_examples_index_name)
        except ImportError:
            raise ImportError("pinecone-client is required but not installed. Install with: pip install pinecone-client")
    
    async def close(self) -> None:
        """Close connections"""
        self.index = None
        self.code_index = None
    
    async def health_check(self) -> bool:
        """Check if Pinecone is responsive"""
        try:
            stats = self.index.describe_index_stats()
            return True
        except Exception as e:
            print(f"Pinecone health check failed: {e}")
            return False
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 100) -> None:
        """Add documents to Pinecone"""
        if not documents:
            return
        
        # Delete existing documents by URL first
        unique_urls = list(set(doc.url for doc in documents))
        await self._delete_by_metadata_filter("url", unique_urls)
        
        # Prepare vectors for insertion
        vectors = []
        for doc in documents:
            vector_id = f"{doc.url}#{doc.chunk_number}"
            vectors.append({
                "id": vector_id,
                "values": doc.embedding,
                "metadata": {
                    "url": doc.url,
                    "chunk_number": doc.chunk_number,
                    "content": doc.content[:1000],  # Pinecone metadata size limit
                    "source_id": doc.source_id,
                    "full_content": doc.content,  # Store full content separately
                    **{k: str(v) for k, v in doc.metadata.items()}  # Convert all to strings
                }
            })
        
        # Insert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            await asyncio.sleep(0.1)  # Rate limiting
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search documents in Pinecone"""
        try:
            filter_dict = {}
            if filter_metadata and "source" in filter_metadata:
                filter_dict["source_id"] = {"$eq": filter_metadata["source"]}
            
            results = self.index.query(
                vector=query_embedding,
                top_k=match_count,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            search_results = []
            for match in results.matches:
                metadata = match.metadata
                search_results.append(VectorSearchResult(
                    id=match.id,
                    content=metadata.get("full_content", metadata.get("content", "")),
                    metadata={k: v for k, v in metadata.items() if k not in ["full_content", "content", "url", "source_id", "chunk_number"]},
                    similarity=match.score,
                    url=metadata.get("url", ""),
                    chunk_number=int(metadata.get("chunk_number", 0)),
                    source_id=metadata.get("source_id", "")
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching documents in Pinecone: {e}")
            return []
    
    async def add_code_examples(self, examples: List[CodeExample], batch_size: int = 100) -> None:
        """Add code examples to Pinecone"""
        if not examples:
            return
        
        # Delete existing code examples by URL first
        unique_urls = list(set(ex.url for ex in examples))
        await self._delete_by_metadata_filter("url", unique_urls, use_code_index=True)
        
        # Prepare vectors for insertion
        vectors = []
        for ex in examples:
            vector_id = f"{ex.url}#{ex.chunk_number}"
            vectors.append({
                "id": vector_id,
                "values": ex.embedding,
                "metadata": {
                    "url": ex.url,
                    "chunk_number": ex.chunk_number,
                    "content": ex.content[:800],  # Pinecone metadata size limit
                    "summary": ex.summary[:200],  # Limit summary size
                    "source_id": ex.source_id,
                    "full_content": ex.content,
                    "full_summary": ex.summary,
                    **{k: str(v) for k, v in ex.metadata.items()}
                }
            })
        
        # Insert in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.code_index.upsert(vectors=batch)
            await asyncio.sleep(0.1)  # Rate limiting
    
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search code examples in Pinecone"""
        try:
            filter_dict = {}
            if filter_metadata and "source" in filter_metadata:
                filter_dict["source_id"] = {"$eq": filter_metadata["source"]}
            
            results = self.code_index.query(
                vector=query_embedding,
                top_k=match_count,
                filter=filter_dict if filter_dict else None,
                include_metadata=True
            )
            
            search_results = []
            for match in results.matches:
                metadata = match.metadata
                search_results.append(VectorSearchResult(
                    id=match.id,
                    content=metadata.get("full_content", metadata.get("content", "")),
                    metadata={k: v for k, v in metadata.items() if k not in ["full_content", "content", "url", "source_id", "chunk_number", "summary", "full_summary"]},
                    similarity=match.score,
                    url=metadata.get("url", ""),
                    chunk_number=int(metadata.get("chunk_number", 0)),
                    source_id=metadata.get("source_id", "")
                ))
            
            return search_results
        except Exception as e:
            print(f"Error searching code examples in Pinecone: {e}")
            return []
    
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources from cache"""
        return [
            SourceInfo(
                source_id=source_id,
                summary=info.get("summary", ""),
                total_words=info.get("total_words", 0),
                created_at=info.get("created_at", ""),
                updated_at=info.get("updated_at", "")
            )
            for source_id, info in self.sources_cache.items()
        ]
    
    async def update_source_info(self, source_id: str, summary: str, word_count: int) -> None:
        """Update source information in cache"""
        import datetime
        
        current_time = datetime.datetime.utcnow().isoformat()
        
        if source_id in self.sources_cache:
            self.sources_cache[source_id].update({
                "summary": summary,
                "total_words": word_count,
                "updated_at": current_time
            })
        else:
            self.sources_cache[source_id] = {
                "summary": summary,
                "total_words": word_count,
                "created_at": current_time,
                "updated_at": current_time
            }
    
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        await self._delete_by_metadata_filter("url", urls)
        await self._delete_by_metadata_filter("url", urls, use_code_index=True)
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        Note: Pinecone doesn't support traditional keyword search,
        so this falls back to vector search only.
        """
        return await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    async def hybrid_search_code_examples(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search for code examples.
        Note: Pinecone doesn't support traditional keyword search,
        so this falls back to vector search only.
        """
        return await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    async def _ensure_indexes_exist(self) -> None:
        """Create indexes if they don't exist"""
        import pinecone
        
        existing_indexes = pinecone.list_indexes()
        
        if self.documents_index_name not in existing_indexes:
            pinecone.create_index(
                name=self.documents_index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
            # Wait for index to be ready
            await asyncio.sleep(10)
        
        if self.code_examples_index_name not in existing_indexes:
            pinecone.create_index(
                name=self.code_examples_index_name,
                dimension=1536,
                metric="cosine"
            )
            # Wait for index to be ready
            await asyncio.sleep(10)
    
    async def _delete_by_metadata_filter(self, key: str, values: List[str], use_code_index: bool = False) -> None:
        """Delete vectors by metadata filter"""
        index = self.code_index if use_code_index else self.index
        
        for value in values:
            filter_dict = {key: {"$eq": value}}
            try:
                index.delete(filter=filter_dict)
                await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error deleting vectors with {key}={value}: {e}")