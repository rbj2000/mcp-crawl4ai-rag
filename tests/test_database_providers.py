import pytest
import asyncio
import os
from typing import List
import tempfile
from src.database.base import VectorDatabaseProvider, DocumentChunk, CodeExample, DatabaseProvider
from src.database.config import DatabaseConfig
from src.database.factory import DatabaseProviderFactory

class DatabaseProviderTestSuite:
    """Comprehensive test suite for all database providers"""
    
    @pytest.fixture
    async def provider(self, provider_config) -> VectorDatabaseProvider:
        """Create and initialize a database provider"""
        provider = DatabaseProviderFactory.create_provider(
            provider_config["type"], 
            provider_config["config"]
        )
        await provider.initialize()
        yield provider
        await provider.close()
    
    async def test_health_check(self, provider):
        """Test database health check"""
        assert await provider.health_check() == True
    
    async def test_document_crud_operations(self, provider):
        """Test basic document CRUD operations"""
        # Create test documents
        test_docs = [
            DocumentChunk(
                url="https://test.com/doc1",
                chunk_number=0,
                content="This is a test document for vector search",
                metadata={"test": True, "category": "docs"},
                embedding=[0.1] * 1536,  # Mock embedding
                source_id="test.com"
            ),
            DocumentChunk(
                url="https://test.com/doc2", 
                chunk_number=0,
                content="Another test document with different content",
                metadata={"test": True, "category": "guides"},
                embedding=[0.2] * 1536,
                source_id="test.com"
            )
        ]
        
        # Add documents
        await provider.add_documents(test_docs)
        
        # Search for documents
        results = await provider.search_documents(
            query_embedding=[0.15] * 1536,
            match_count=2
        )
        
        assert len(results) >= 1
        assert any("test document" in result.content for result in results)
        
        # Test filtering
        filtered_results = await provider.search_documents(
            query_embedding=[0.15] * 1536,
            match_count=2,
            filter_metadata={"source": "test.com"}
        )
        
        assert len(filtered_results) >= 1
        assert all(result.source_id == "test.com" for result in filtered_results)
    
    async def test_code_example_operations(self, provider):
        """Test code example operations"""
        test_examples = [
            CodeExample(
                url="https://test.com/code1",
                chunk_number=0,
                content="def hello_world():\n    print('Hello, World!')",
                summary="A simple hello world function",
                metadata={"language": "python", "test": True},
                embedding=[0.3] * 1536,
                source_id="test.com"
            )
        ]
        
        await provider.add_code_examples(test_examples)
        
        results = await provider.search_code_examples(
            query_embedding=[0.35] * 1536,
            match_count=1
        )
        
        assert len(results) >= 1
        assert "hello_world" in results[0].content
    
    async def test_source_management(self, provider):
        """Test source information management"""
        await provider.update_source_info(
            source_id="test.com",
            summary="Test website for unit tests",
            word_count=100
        )
        
        sources = await provider.get_sources()
        test_source = next((s for s in sources if s.source_id == "test.com"), None)
        
        assert test_source is not None
        assert test_source.summary == "Test website for unit tests"
        assert test_source.total_words == 100
    
    async def test_hybrid_search(self, provider):
        """Test hybrid search if supported"""
        if hasattr(provider, 'hybrid_search_documents'):
            # Add test documents first
            test_docs = [
                DocumentChunk(
                    url="https://test.com/hybrid1",
                    chunk_number=0,
                    content="Python is a programming language with great vector support",
                    metadata={"test": True},
                    embedding=[0.4] * 1536,
                    source_id="test.com"
                )
            ]
            
            await provider.add_documents(test_docs)
            
            # Test hybrid search
            results = await provider.hybrid_search_documents(
                query_embedding=[0.45] * 1536,
                keyword_query="Python programming",
                match_count=1
            )
            
            assert len(results) >= 1
            assert "Python" in results[0].content

@pytest.mark.asyncio
class TestSQLiteProvider(DatabaseProviderTestSuite):
    """Test SQLite provider"""
    
    @pytest.fixture
    def provider_config(self):
        # Use a temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.sqlite')
        temp_file.close()
        
        return {
            "type": DatabaseProvider.SQLITE,
            "config": {
                "db_path": temp_file.name,
                "embedding_dimension": 1536
            }
        }

# Add tests for other providers when their dependencies are available
if os.getenv("TEST_SUPABASE_URL") and os.getenv("TEST_SUPABASE_SERVICE_KEY"):
    @pytest.mark.asyncio
    class TestSupabaseProvider(DatabaseProviderTestSuite):
        """Test Supabase provider"""
        
        @pytest.fixture
        def provider_config(self):
            return {
                "type": DatabaseProvider.SUPABASE,
                "config": {
                    "url": os.getenv("TEST_SUPABASE_URL"),
                    "service_key": os.getenv("TEST_SUPABASE_SERVICE_KEY")
                }
            }

if os.getenv("TEST_PINECONE_API_KEY"):
    @pytest.mark.asyncio
    class TestPineconeProvider(DatabaseProviderTestSuite):
        """Test Pinecone provider"""
        
        @pytest.fixture
        def provider_config(self):
            return {
                "type": DatabaseProvider.PINECONE,
                "config": {
                    "api_key": os.getenv("TEST_PINECONE_API_KEY"),
                    "environment": os.getenv("TEST_PINECONE_ENVIRONMENT"),
                    "index_name": "test-crawl4ai-rag"
                }
            }

if os.getenv("TEST_NEO4J_URI"):
    @pytest.mark.asyncio
    class TestNeo4jProvider(DatabaseProviderTestSuite):
        """Test Neo4j provider"""
        
        @pytest.fixture
        def provider_config(self):
            return {
                "type": DatabaseProvider.NEO4J_VECTOR,
                "config": {
                    "uri": os.getenv("TEST_NEO4J_URI"),
                    "user": os.getenv("TEST_NEO4J_USER", "neo4j"),
                    "password": os.getenv("TEST_NEO4J_PASSWORD"),
                    "database": os.getenv("TEST_NEO4J_DATABASE", "neo4j")
                }
            }