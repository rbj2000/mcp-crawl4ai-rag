import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

logger = logging.getLogger(__name__)

class Neo4jVectorProvider(VectorDatabaseProvider):
    """Neo4j vector database implementation using vector indexes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
        self.database = config.get("database", "neo4j")
        self.embedding_dimension = config.get("embedding_dimension", 1536)
        logger.info(f"Neo4j provider initialized with embedding dimension: {self.embedding_dimension}")
    
    def get_embedding_dimension(self) -> int:
        """Get the configured embedding dimension for this provider"""
        return self.embedding_dimension
    
    async def validate_embedding_dimension(self, dimension: int) -> bool:
        """Validate if the given embedding dimension is compatible"""
        try:
            async with self.driver.session(database=self.database) as session:
                # Check if we have any existing vector indexes
                result = await session.run("""
                    SHOW INDEXES 
                    WHERE type = 'VECTOR' AND name IN ['document_embeddings', 'code_embeddings']
                """)
                
                indexes = []
                async for record in result:
                    indexes.append(record)
                
                if not indexes:
                    return True  # No existing vector indexes
                
                # Check if existing indexes have compatible dimensions
                for index in indexes:
                    # This is a simplified check - in practice, you'd need to query
                    # the index configuration to get the actual dimension
                    if hasattr(index, 'options') and 'vector.dimensions' in index.options:
                        existing_dim = int(index.options['vector.dimensions'])
                        if existing_dim != dimension:
                            logger.error(
                                f"Vector index dimension mismatch: existing {existing_dim}, "
                                f"requested {dimension}"
                            )
                            return False
                
                return True
        except Exception as e:
            logger.error(f"Error validating embedding dimension: {e}")
            return False
    
    async def migrate_dimension(self, new_dimension: int) -> bool:
        """Migrate to a new embedding dimension"""
        logger.warning(
            f"Neo4j provider cannot automatically migrate from dimension "
            f"{self.embedding_dimension} to {new_dimension}. "
            f"You need to recreate vector indexes and regenerate embeddings."
        )
        
        # Provide migration guidance
        logger.info(
            f"Migration steps:\n"
            f"1. Drop existing vector indexes: DROP INDEX document_embeddings; DROP INDEX code_embeddings;\n"
            f"2. Recreate vector indexes with new dimension ({new_dimension})\n"
            f"3. Delete existing nodes: MATCH (d:Document), (c:CodeExample) DELETE d, c;\n"
            f"4. Regenerate and re-insert all embeddings with the new dimension"
        )
        
        return False

    async def initialize(self) -> None:
        """Initialize Neo4j connection and create vector indexes"""
        try:
            from neo4j import AsyncGraphDatabase
            
            self.driver = AsyncGraphDatabase.driver(
                self.config["uri"],
                auth=(self.config["user"], self.config["password"])
            )
            
            # Verify connection
            await self.health_check()
            
            # Create vector indexes and constraints
            await self._create_schema()
            
        except ImportError:
            raise ImportError("neo4j is required but not installed. Install with: pip install neo4j")
    
    async def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            self.driver = None
    
    async def health_check(self) -> bool:
        """Check if Neo4j is responsive"""
        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
                return True
        except Exception as e:
            print(f"Neo4j health check failed: {e}")
            return False
    
    async def _create_schema(self) -> None:
        """Create necessary indexes and constraints"""
        async with self.driver.session(database=self.database) as session:
            
            # Create constraints for unique identification
            constraints = [
                "CREATE CONSTRAINT document_unique IF NOT EXISTS FOR (d:Document) REQUIRE (d.url, d.chunk_number) IS UNIQUE",
                "CREATE CONSTRAINT code_example_unique IF NOT EXISTS FOR (c:CodeExample) REQUIRE (c.url, c.chunk_number) IS UNIQUE",
                "CREATE CONSTRAINT source_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    print(f"Constraint creation note: {e}")
            
            # Create vector indexes for similarity search
            vector_indexes = [
                f"""
                CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                FOR (d:Document) ON (d.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {self.embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                f"""
                CREATE VECTOR INDEX code_embeddings IF NOT EXISTS
                FOR (c:CodeExample) ON (c.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {self.embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            ]
            
            for index in vector_indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    print(f"Vector index creation note: {e}")
            
            # Create regular indexes for filtering
            regular_indexes = [
                "CREATE INDEX document_source_id IF NOT EXISTS FOR (d:Document) ON (d.source_id)",
                "CREATE INDEX document_url IF NOT EXISTS FOR (d:Document) ON (d.url)",
                "CREATE INDEX code_source_id IF NOT EXISTS FOR (c:CodeExample) ON (c.source_id)",
                "CREATE INDEX code_url IF NOT EXISTS FOR (c:CodeExample) ON (c.url)"
            ]
            
            for index in regular_indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    print(f"Regular index creation note: {e}")
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        """Add documents to Neo4j"""
        if not documents:
            return
        
        # Validate embedding dimensions
        for doc in documents:
            if len(doc.embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Document embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(doc.embedding)} for URL {doc.url}"
                )
        
        async with self.driver.session(database=self.database) as session:
            # Delete existing documents by URL
            unique_urls = list(set(doc.url for doc in documents))
            
            await session.run(
                "UNWIND $urls AS url MATCH (d:Document {url: url}) DELETE d",
                urls=unique_urls
            )
            
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                batch_documents = documents[i:batch_end]
                
                # Prepare batch data
                batch_data = []
                for doc in batch_documents:
                    batch_data.append({
                        "url": doc.url,
                        "chunk_number": doc.chunk_number,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "source_id": doc.source_id,
                        "embedding": doc.embedding
                    })
                
                # Insert batch
                await session.run("""
                    UNWIND $batch AS doc
                    CREATE (d:Document {
                        url: doc.url,
                        chunk_number: doc.chunk_number,
                        content: doc.content,
                        metadata: doc.metadata,
                        source_id: doc.source_id,
                        embedding: doc.embedding,
                        created_at: datetime()
                    })
                """, batch=batch_data)
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search documents using vector similarity"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        async with self.driver.session(database=self.database) as session:
            
            # Build query with optional filtering
            query = """
                CALL db.index.vector.queryNodes('document_embeddings', $match_count, $query_embedding)
                YIELD node AS d, score
            """
            
            params = {
                "query_embedding": query_embedding,
                "match_count": match_count
            }
            
            # Add source filtering if provided
            if filter_metadata and "source" in filter_metadata:
                query += " WHERE d.source_id = $source_filter"
                params["source_filter"] = filter_metadata["source"]
            
            query += """
                RETURN d.url as url, d.chunk_number as chunk_number, d.content as content,
                       d.metadata as metadata, d.source_id as source_id, score,
                       elementId(d) as id
                ORDER BY score DESC
            """
            
            result = await session.run(query, **params)
            
            search_results = []
            async for record in result:
                search_results.append(VectorSearchResult(
                    id=record["id"],
                    content=record["content"],
                    metadata=record["metadata"] or {},
                    similarity=float(record["score"]),
                    url=record["url"],
                    chunk_number=record["chunk_number"],
                    source_id=record["source_id"]
                ))
            
            return search_results
    
    async def add_code_examples(self, examples: List[CodeExample], batch_size: int = 20) -> None:
        """Add code examples to Neo4j"""
        if not examples:
            return
        
        # Validate embedding dimensions
        for ex in examples:
            if len(ex.embedding) != self.embedding_dimension:
                raise ValueError(
                    f"Code example embedding dimension mismatch: expected {self.embedding_dimension}, "
                    f"got {len(ex.embedding)} for URL {ex.url}"
                )
        
        async with self.driver.session(database=self.database) as session:
            # Delete existing code examples by URL
            unique_urls = list(set(ex.url for ex in examples))
            
            await session.run(
                "UNWIND $urls AS url MATCH (c:CodeExample {url: url}) DELETE c",
                urls=unique_urls
            )
            
            # Process in batches
            for i in range(0, len(examples), batch_size):
                batch_end = min(i + batch_size, len(examples))
                batch_examples = examples[i:batch_end]
                
                # Prepare batch data
                batch_data = []
                for ex in batch_examples:
                    batch_data.append({
                        "url": ex.url,
                        "chunk_number": ex.chunk_number,
                        "content": ex.content,
                        "summary": ex.summary,
                        "metadata": ex.metadata,
                        "source_id": ex.source_id,
                        "embedding": ex.embedding
                    })
                
                # Insert batch
                await session.run("""
                    UNWIND $batch AS ex
                    CREATE (c:CodeExample {
                        url: ex.url,
                        chunk_number: ex.chunk_number,
                        content: ex.content,
                        summary: ex.summary,
                        metadata: ex.metadata,
                        source_id: ex.source_id,
                        embedding: ex.embedding,
                        created_at: datetime()
                    })
                """, batch=batch_data)
    
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search code examples using vector similarity"""
        # Validate embedding dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(query_embedding)}"
            )
        
        async with self.driver.session(database=self.database) as session:
            
            # Build query with optional filtering
            query = """
                CALL db.index.vector.queryNodes('code_embeddings', $match_count, $query_embedding)
                YIELD node AS c, score
            """
            
            params = {
                "query_embedding": query_embedding,
                "match_count": match_count
            }
            
            # Add source filtering if provided
            if filter_metadata and "source" in filter_metadata:
                query += " WHERE c.source_id = $source_filter"
                params["source_filter"] = filter_metadata["source"]
            
            query += """
                RETURN c.url as url, c.chunk_number as chunk_number, c.content as content,
                       c.summary as summary, c.metadata as metadata, c.source_id as source_id, score,
                       elementId(c) as id
                ORDER BY score DESC
            """
            
            result = await session.run(query, **params)
            
            search_results = []
            async for record in result:
                search_results.append(VectorSearchResult(
                    id=record["id"],
                    content=record["content"],
                    metadata=record["metadata"] or {},
                    similarity=float(record["score"]),
                    url=record["url"],
                    chunk_number=record["chunk_number"],
                    source_id=record["source_id"]
                ))
            
            return search_results
    
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        async with self.driver.session(database=self.database) as session:
            result = await session.run("""
                MATCH (s:Source)
                RETURN s.source_id as source_id, s.summary as summary, 
                       s.total_words as total_words, s.created_at as created_at,
                       s.updated_at as updated_at
                ORDER BY s.source_id
            """)
            
            sources = []
            async for record in result:
                sources.append(SourceInfo(
                    source_id=record["source_id"],
                    summary=record["summary"] or "",
                    total_words=record["total_words"] or 0,
                    created_at=str(record["created_at"]) if record["created_at"] else "",
                    updated_at=str(record["updated_at"]) if record["updated_at"] else ""
                ))
            
            return sources
    
    async def update_source_info(self, source_id: str, summary: str, word_count: int) -> None:
        """Update source information"""
        async with self.driver.session(database=self.database) as session:
            await session.run("""
                MERGE (s:Source {source_id: $source_id})
                SET s.summary = $summary,
                    s.total_words = $word_count,
                    s.updated_at = datetime()
                ON CREATE SET s.created_at = datetime()
            """, source_id=source_id, summary=summary, word_count=word_count)
    
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        async with self.driver.session(database=self.database) as session:
            # Delete documents
            await session.run("""
                UNWIND $urls AS url
                MATCH (d:Document {url: url})
                DELETE d
            """, urls=urls)
            
            # Delete code examples
            await session.run("""
                UNWIND $urls AS url
                MATCH (c:CodeExample {url: url})
                DELETE c
            """, urls=urls)
    
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
        
        async with self.driver.session(database=self.database) as session:
            
            # Get vector search results
            vector_results = await self.search_documents(
                query_embedding=query_embedding,
                match_count=match_count * 2,
                filter_metadata=filter_metadata
            )
            
            # Get keyword search results
            keyword_query_cypher = """
                MATCH (d:Document)
                WHERE d.content CONTAINS $keyword_query
            """
            
            params = {"keyword_query": keyword_query}
            
            # Apply source filtering if provided
            if filter_metadata and "source" in filter_metadata:
                keyword_query_cypher += " AND d.source_id = $source_filter"
                params["source_filter"] = filter_metadata["source"]
            
            keyword_query_cypher += """
                RETURN d.url as url, d.chunk_number as chunk_number, d.content as content,
                       d.metadata as metadata, d.source_id as source_id,
                       elementId(d) as id
                LIMIT $limit
            """
            params["limit"] = match_count * 2
            
            result = await session.run(keyword_query_cypher, **params)
            
            keyword_results = []
            async for record in result:
                keyword_results.append(VectorSearchResult(
                    id=record["id"],
                    content=record["content"],
                    metadata=record["metadata"] or {},
                    similarity=0.5,  # Default similarity for keyword matches
                    url=record["url"],
                    chunk_number=record["chunk_number"],
                    source_id=record["source_id"]
                ))
            
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
        
        async with self.driver.session(database=self.database) as session:
            
            # Get vector search results
            vector_results = await self.search_code_examples(
                query_embedding=query_embedding,
                match_count=match_count * 2,
                filter_metadata=filter_metadata
            )
            
            # Get keyword search results
            keyword_query_cypher = """
                MATCH (c:CodeExample)
                WHERE c.content CONTAINS $keyword_query OR c.summary CONTAINS $keyword_query
            """
            
            params = {"keyword_query": keyword_query}
            
            # Apply source filtering if provided
            if filter_metadata and "source" in filter_metadata:
                keyword_query_cypher += " AND c.source_id = $source_filter"
                params["source_filter"] = filter_metadata["source"]
            
            keyword_query_cypher += """
                RETURN c.url as url, c.chunk_number as chunk_number, c.content as content,
                       c.summary as summary, c.metadata as metadata, c.source_id as source_id,
                       elementId(c) as id
                LIMIT $limit
            """
            params["limit"] = match_count * 2
            
            result = await session.run(keyword_query_cypher, **params)
            
            keyword_results = []
            async for record in result:
                keyword_results.append(VectorSearchResult(
                    id=record["id"],
                    content=record["content"],
                    metadata=record["metadata"] or {},
                    similarity=0.5,  # Default similarity for keyword matches
                    url=record["url"],
                    chunk_number=record["chunk_number"],
                    source_id=record["source_id"]
                ))
            
            # Combine results
            return self._combine_hybrid_results(vector_results, keyword_results, match_count)
    
    def _combine_hybrid_results(
        self, 
        vector_results: List[VectorSearchResult], 
        keyword_results: List[VectorSearchResult], 
        match_count: int
    ) -> List[VectorSearchResult]:
        """Combine vector and keyword search results"""
        seen_ids = set()
        combined_results = []
        
        # Boost items that appear in both results
        vector_ids = {r.id for r in vector_results}
        for kr in keyword_results:
            if kr.id in vector_ids and kr.id not in seen_ids:
                # Find the vector result and boost its score
                for vr in vector_results:
                    if vr.id == kr.id:
                        vr.similarity = min(1.0, vr.similarity * 1.2)
                        combined_results.append(vr)
                        seen_ids.add(kr.id)
                        break
        
        # Add remaining vector results
        for vr in vector_results:
            if vr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(vr)
                seen_ids.add(vr.id)
        
        # Add remaining keyword results
        for kr in keyword_results:
            if kr.id not in seen_ids and len(combined_results) < match_count:
                combined_results.append(kr)
                seen_ids.add(kr.id)
        
        return combined_results[:match_count]