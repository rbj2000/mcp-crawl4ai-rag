from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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
    embedding_dimension: Optional[int] = None
    
    def __post_init__(self):
        """Validate embedding dimensions"""
        if self.embedding_dimension is None:
            self.embedding_dimension = len(self.embedding)
        elif len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(self.embedding)}"
            )

@dataclass
class CodeExample:
    url: str
    chunk_number: int
    content: str
    summary: str
    metadata: Dict[str, Any]
    embedding: List[float]
    source_id: str
    embedding_dimension: Optional[int] = None
    
    def __post_init__(self):
        """Validate embedding dimensions"""
        if self.embedding_dimension is None:
            self.embedding_dimension = len(self.embedding)
        elif len(self.embedding) != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {len(self.embedding)}"
            )

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
    def get_embedding_dimension(self) -> int:
        """Get the configured embedding dimension for this provider
        
        Returns:
            int: The embedding dimension used by this provider
        """
        pass
    
    @abstractmethod
    async def validate_embedding_dimension(self, dimension: int) -> bool:
        """Validate if the given embedding dimension is compatible
        
        Args:
            dimension: The embedding dimension to validate
            
        Returns:
            bool: True if dimension is compatible, False otherwise
        """
        pass
    
    @abstractmethod
    async def migrate_dimension(self, new_dimension: int) -> bool:
        """Migrate to a new embedding dimension
        
        Args:
            new_dimension: The new embedding dimension to migrate to
            
        Returns:
            bool: True if migration was successful, False otherwise
            
        Note:
            This method should handle data migration or provide guidance
            for manual migration when automatic migration is not possible.
        """
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
        """Search for similar documents
        
        Args:
            query_embedding: The query embedding vector
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ValueError: If query_embedding dimension doesn't match configured dimension
        """
    
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
        """Search for similar code examples
        
        Args:
            query_embedding: The query embedding vector
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ValueError: If query_embedding dimension doesn't match configured dimension
        """
    
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
        
        Args:
            query_embedding: The query embedding vector
            keyword_query: The keyword search query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ValueError: If query_embedding dimension doesn't match configured dimension
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
        
        Args:
            query_embedding: The query embedding vector
            keyword_query: The keyword search query
            match_count: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of VectorSearchResult objects
            
        Raises:
            ValueError: If query_embedding dimension doesn't match configured dimension
        """
        
        return await self.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )