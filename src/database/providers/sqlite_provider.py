import sqlite3
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from ..base import (
    VectorDatabaseProvider, 
    DocumentChunk, 
    CodeExample, 
    VectorSearchResult, 
    SourceInfo
)

class SQLiteProvider(VectorDatabaseProvider):
    """Local SQLite with vector similarity implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config["db_path"]
        self.embedding_dimension = config.get("embedding_dimension", 1536)
        self.connection = None
    
    async def initialize(self) -> None:
        """Initialize SQLite database"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        await self._create_tables()
        await self._create_indexes()
    
    async def close(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def health_check(self) -> bool:
        """Check if database is responsive"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"SQLite health check failed: {e}")
            return False
    
    async def _create_tables(self) -> None:
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Code examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, chunk_number)
            )
        """)
        
        # Sources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                summary TEXT,
                total_words INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    async def _create_indexes(self) -> None:
        """Create database indexes"""
        cursor = self.connection.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url)",
            "CREATE INDEX IF NOT EXISTS idx_documents_source_id ON documents(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_code_examples_url ON code_examples(url)",
            "CREATE INDEX IF NOT EXISTS idx_code_examples_source_id ON code_examples(source_id)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
    
    async def add_documents(self, documents: List[DocumentChunk], batch_size: int = 20) -> None:
        """Add documents to SQLite"""
        if not documents:
            return
        
        cursor = self.connection.cursor()
        
        # Delete existing documents by URL
        unique_urls = list(set(doc.url for doc in documents))
        placeholders = ",".join("?" * len(unique_urls))
        cursor.execute(f"DELETE FROM documents WHERE url IN ({placeholders})", unique_urls)
        
        # Prepare data for insertion
        data = []
        for doc in documents:
            embedding_blob = np.array(doc.embedding, dtype=np.float32).tobytes()
            data.append((
                doc.url,
                doc.chunk_number,
                doc.content,
                json.dumps(doc.metadata),
                doc.source_id,
                embedding_blob
            ))
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany("""
                INSERT OR REPLACE INTO documents 
                (url, chunk_number, content, metadata, source_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
            """, batch)
        
        self.connection.commit()
    
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search documents using cosine similarity"""
        cursor = self.connection.cursor()
        
        # Build query with optional filtering
        sql = """
            SELECT id, url, chunk_number, content, metadata, source_id, embedding
            FROM documents
        """
        params = []
        
        if filter_metadata and "source" in filter_metadata:
            sql += " WHERE source_id = ?"
            params.append(filter_metadata["source"])
        
        cursor.execute(sql, params)
        
        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        for row in cursor.fetchall():
            # Deserialize embedding
            doc_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_vec], [doc_embedding])[0][0]
            
            results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=float(similarity),
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
            ))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:match_count]
    
    async def add_code_examples(self, examples: List[CodeExample], batch_size: int = 20) -> None:
        """Add code examples to SQLite"""
        if not examples:
            return
        
        cursor = self.connection.cursor()
        
        # Delete existing code examples by URL
        unique_urls = list(set(ex.url for ex in examples))
        placeholders = ",".join("?" * len(unique_urls))
        cursor.execute(f"DELETE FROM code_examples WHERE url IN ({placeholders})", unique_urls)
        
        # Prepare data for insertion
        data = []
        for ex in examples:
            embedding_blob = np.array(ex.embedding, dtype=np.float32).tobytes()
            data.append((
                ex.url,
                ex.chunk_number,
                ex.content,
                ex.summary,
                json.dumps(ex.metadata),
                ex.source_id,
                embedding_blob
            ))
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            cursor.executemany("""
                INSERT OR REPLACE INTO code_examples 
                (url, chunk_number, content, summary, metadata, source_id, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, batch)
        
        self.connection.commit()
    
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search code examples using cosine similarity"""
        cursor = self.connection.cursor()
        
        # Build query with optional filtering
        sql = """
            SELECT id, url, chunk_number, content, summary, metadata, source_id, embedding
            FROM code_examples
        """
        params = []
        
        if filter_metadata and "source" in filter_metadata:
            sql += " WHERE source_id = ?"
            params.append(filter_metadata["source"])
        
        cursor.execute(sql, params)
        
        results = []
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        for row in cursor.fetchall():
            # Deserialize embedding
            code_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_vec], [code_embedding])[0][0]
            
            results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=float(similarity),
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
            ))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:match_count]
    
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM sources ORDER BY source_id")
            
            results = []
            for row in cursor.fetchall():
                results.append(SourceInfo(
                    source_id=row["source_id"],
                    summary=row["summary"],
                    total_words=row["total_words"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                ))
            
            return results
        except Exception as e:
            print(f"Error getting sources: {e}")
            return []
    
    async def update_source_info(self, source_id: str, summary: str, word_count: int) -> None:
        """Update source information"""
        try:
            cursor = self.connection.cursor()
            
            # Try to update existing source
            cursor.execute("""
                UPDATE sources 
                SET summary = ?, total_words = ?, updated_at = CURRENT_TIMESTAMP
                WHERE source_id = ?
            """, (summary, word_count, source_id))
            
            # If no rows were updated, insert new source
            if cursor.rowcount == 0:
                cursor.execute("""
                    INSERT INTO sources (source_id, summary, total_words)
                    VALUES (?, ?, ?)
                """, (source_id, summary, word_count))
            
            self.connection.commit()
        except Exception as e:
            print(f"Error updating source {source_id}: {e}")
    
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        try:
            cursor = self.connection.cursor()
            placeholders = ",".join("?" * len(urls))
            
            # Delete from both tables
            cursor.execute(f"DELETE FROM documents WHERE url IN ({placeholders})", urls)
            cursor.execute(f"DELETE FROM code_examples WHERE url IN ({placeholders})", urls)
            
            self.connection.commit()
        except Exception as e:
            print(f"Error deleting by URLs: {e}")
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Hybrid search combining vector similarity and keyword search"""
        # Get vector search results
        vector_results = await self.search_documents(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        cursor = self.connection.cursor()
        sql = """
            SELECT id, url, chunk_number, content, metadata, source_id
            FROM documents
            WHERE content LIKE ?
        """
        params = [f"%{keyword_query}%"]
        
        if filter_metadata and "source" in filter_metadata:
            sql += " AND source_id = ?"
            params.append(filter_metadata["source"])
        
        sql += " LIMIT ?"
        params.append(match_count * 2)
        
        cursor.execute(sql, params)
        
        keyword_results = []
        for row in cursor.fetchall():
            keyword_results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=0.5,  # Default similarity for keyword matches
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
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
        """Hybrid search for code examples"""
        # Get vector search results
        vector_results = await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count * 2,
            filter_metadata=filter_metadata
        )
        
        # Get keyword search results
        cursor = self.connection.cursor()
        sql = """
            SELECT id, url, chunk_number, content, summary, metadata, source_id
            FROM code_examples
            WHERE content LIKE ? OR summary LIKE ?
        """
        params = [f"%{keyword_query}%", f"%{keyword_query}%"]
        
        if filter_metadata and "source" in filter_metadata:
            sql += " AND source_id = ?"
            params.append(filter_metadata["source"])
        
        sql += " LIMIT ?"
        params.append(match_count * 2)
        
        cursor.execute(sql, params)
        
        keyword_results = []
        for row in cursor.fetchall():
            keyword_results.append(VectorSearchResult(
                id=str(row["id"]),
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                similarity=0.5,  # Default similarity for keyword matches
                url=row["url"],
                chunk_number=row["chunk_number"],
                source_id=row["source_id"]
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