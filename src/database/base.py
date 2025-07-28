from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class DatabaseProvider(Enum):
    SUPABASE = "supabase"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    SQLITE = "sqlite"
    NEO4J_VECTOR = "neo4j_vector"

@dataclass
class VectorSearchResult:
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    url: str
    chunk_number: int
    source_id: str

@dataclass
class DocumentChunk:
    url: str
    chunk_number: int
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]
    source_id: str

@dataclass
class CodeExample:
    url: str
    chunk_number: int
    content: str
    summary: str
    metadata: Dict[str, Any]
    embedding: List[float]
    source_id: str

@dataclass
class SourceInfo:
    source_id: str
    summary: str
    total_words: int
    created_at: str
    updated_at: str

class VectorDatabaseProvider(ABC):
    """Abstract base class for vector database providers"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database connection and schema"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the database connection"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the database is healthy and responsive"""
        pass
    
    @abstractmethod
    async def add_documents(
        self, 
        documents: List[DocumentChunk], 
        batch_size: int = 20
    ) -> None:
        """Add document chunks to the database"""
        pass
    
    @abstractmethod
    async def search_documents(
        self, 
        query_embedding: List[float], 
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    async def add_code_examples(
        self, 
        examples: List[CodeExample], 
        batch_size: int = 20
    ) -> None:
        """Add code examples to the database"""
        pass
    
    @abstractmethod
    async def search_code_examples(
        self,
        query_embedding: List[float],
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar code examples"""
        pass
    
    @abstractmethod
    async def get_sources(self) -> List[SourceInfo]:
        """Get all available sources"""
        pass
    
    @abstractmethod
    async def update_source_info(
        self, 
        source_id: str, 
        summary: str, 
        word_count: int
    ) -> None:
        """Update source information"""
        pass
    
    @abstractmethod
    async def delete_by_urls(self, urls: List[str]) -> None:
        """Delete documents by URLs"""
        pass
    
    async def hybrid_search_documents(
        self,
        query_embedding: List[float],
        keyword_query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        Default implementation falls back to vector search only.
        Providers can override this for true hybrid search.
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
        Default implementation falls back to vector search only.
        """
        return await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )