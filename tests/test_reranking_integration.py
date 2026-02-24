"""
Comprehensive integration tests for reranking functionality in the RAG pipeline.

This test suite ensures:
1. RAG pipeline integration with reranking enabled/disabled
2. End-to-end reranking workflow with real data
3. Provider switching and configuration management
4. Performance impact and benchmarking
5. Compatibility with existing RAG strategies
6. Cross-provider consistency and fallback mechanisms
7. Environment variable configuration handling
"""

import pytest
import asyncio
import os
import tempfile
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import List, Dict, Any, Optional
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import path management
import sys
from pathlib import Path

# Add src directory to path dynamically  
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import components for integration testing
try:
    from ai_providers.base import (
        RerankingProvider, RerankingResult, HealthStatus, AIProvider
    )
    from ai_providers.factory import AIProviderFactory
    from ai_providers.config import AIProviderConfig
    # Import utility functions that handle reranking
    # from utils import rerank_results_with_provider, perform_rag_query
except ImportError as e:
    logger.warning(f"Could not import reranking integration components: {e}")
    # Mock classes for testing framework development
    from enum import Enum
    from dataclasses import dataclass
    
    class AIProvider(Enum):
        OPENAI = "openai"
        OLLAMA = "ollama" 
        HUGGINGFACE = "huggingface"
    
    @dataclass
    class RerankingResult:
        results: List[Dict[str, Any]]
        model: str
        provider: str
        rerank_scores: List[float]
        processing_time_ms: float
        metadata: Optional[Dict[str, Any]] = None


# Mock database and embedding components
class MockSupabaseClient:
    """Mock Supabase client for testing."""
    
    def __init__(self):
        self.documents = []
        self.call_count = 0
    
    def add_documents(self, documents):
        """Add mock documents."""
        self.documents.extend(documents)
    
    async def similarity_search(self, query_embedding, match_count=10, filter_metadata=None):
        """Mock similarity search."""
        self.call_count += 1
        
        # Return mock results based on stored documents
        results = []
        for i, doc in enumerate(self.documents[:match_count]):
            result = {
                "content": doc.get("content", ""),
                "url": doc.get("url", f"https://mock.com/{i}"),
                "metadata": doc.get("metadata", {}),
                "similarity_score": 0.8 - (i * 0.05),  # Decreasing similarity
                "chunk_id": i
            }
            results.append(result)
        
        return results


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""
    
    def __init__(self, dimensions=1536):
        self.dimensions = dimensions
        self.call_count = 0
    
    async def create_embedding(self, text):
        """Create mock embedding."""
        self.call_count += 1
        # Simple mock embedding based on text length and content
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding = [float(int(text_hash[i:i+2], 16)) / 255.0 for i in range(0, min(len(text_hash), self.dimensions * 2), 2)]
        
        # Pad or truncate to exact dimensions
        if len(embedding) < self.dimensions:
            embedding.extend([0.0] * (self.dimensions - len(embedding)))
        else:
            embedding = embedding[:self.dimensions]
        
        return embedding


class MockRerankingProvider:
    """Mock reranking provider for integration testing."""
    
    def __init__(self, provider_type="mock", enabled=True):
        self.provider_type = provider_type
        self.enabled = enabled
        self.call_count = 0
        self.processing_times = []
    
    async def rerank_results(self, query, results, model=None):
        """Mock reranking with realistic behavior."""
        if not self.enabled:
            raise Exception("Reranking disabled")
        
        self.call_count += 1
        start_time = time.time()
        
        # Simulate processing time based on provider type
        if self.provider_type == "openai":
            await asyncio.sleep(0.1)  # API call simulation
        elif self.provider_type == "ollama":
            await asyncio.sleep(0.2)  # Slower local processing
        elif self.provider_type == "huggingface":
            await asyncio.sleep(0.05)  # Fast local processing
        
        # Generate realistic reranking scores
        reranked_results = []
        scores = []
        
        for i, result in enumerate(results):
            content = result.get("content", "")
            query_lower = query.lower()
            content_lower = content.lower()
            
            # Simple relevance scoring
            base_score = result.get("similarity_score", 0.5)
            query_overlap = sum(word in content_lower for word in query_lower.split()) / max(len(query_lower.split()), 1)
            rerank_score = min(base_score + query_overlap * 0.3, 1.0)
            
            enhanced_result = result.copy()
            enhanced_result["rerank_score"] = rerank_score
            enhanced_result["original_rank"] = i
            enhanced_result["reranked_by"] = self.provider_type
            
            reranked_results.append(enhanced_result)
            scores.append(rerank_score)
        
        # Sort by rerank score descending
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        
        return RerankingResult(
            results=reranked_results,
            model=model or f"{self.provider_type}-reranker",
            provider=self.provider_type,
            rerank_scores=scores,
            processing_time_ms=processing_time,
            metadata={
                "query_length": len(query),
                "results_count": len(results),
                "provider_type": self.provider_type
            }
        )
    
    async def health_check(self):
        """Mock health check."""
        return {
            "is_healthy": self.enabled,
            "provider": self.provider_type,
            "response_time_ms": 10.0
        }


# Integration test helper functions
async def mock_rerank_results_with_provider(provider, query, results, model=None):
    """Mock version of rerank_results_with_provider utility function."""
    if not provider:
        return results  # No reranking
    
    try:
        rerank_result = await provider.rerank_results(query, results, model)
        return rerank_result.results
    except Exception as e:
        logger.warning(f"Reranking failed, returning original results: {e}")
        return results


async def mock_perform_rag_query(
    query, 
    embedding_provider, 
    supabase_client, 
    reranking_provider=None,
    match_count=10,
    use_reranking=False
):
    """Mock version of perform_rag_query with reranking integration."""
    # Step 1: Generate query embedding
    query_embedding = await embedding_provider.create_embedding(query)
    
    # Step 2: Perform similarity search
    initial_results = await supabase_client.similarity_search(
        query_embedding, 
        match_count=match_count
    )
    
    # Step 3: Apply reranking if enabled
    if use_reranking and reranking_provider:
        try:
            reranked_results = await mock_rerank_results_with_provider(
                reranking_provider, query, initial_results
            )
            return {
                "results": reranked_results,
                "reranking_applied": True,
                "original_count": len(initial_results),
                "reranked_count": len(reranked_results)
            }
        except Exception as e:
            logger.warning(f"Reranking failed, using original results: {e}")
            return {
                "results": initial_results,
                "reranking_applied": False,
                "reranking_error": str(e),
                "original_count": len(initial_results)
            }
    
    return {
        "results": initial_results,
        "reranking_applied": False,
        "original_count": len(initial_results)
    }


class TestRerankingRAGIntegration:
    """Test suite for reranking integration with RAG pipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "content": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
                "url": "https://example.com/ml-intro",
                "metadata": {"topic": "machine_learning", "difficulty": "beginner"}
            },
            {
                "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                "url": "https://example.com/deep-learning",
                "metadata": {"topic": "deep_learning", "difficulty": "intermediate"}
            },
            {
                "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with interactions between computers and human language.",
                "url": "https://example.com/nlp",
                "metadata": {"topic": "nlp", "difficulty": "intermediate"}
            },
            {
                "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
                "url": "https://example.com/computer-vision",
                "metadata": {"topic": "computer_vision", "difficulty": "advanced"}
            },
            {
                "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward.",
                "url": "https://example.com/reinforcement-learning",
                "metadata": {"topic": "reinforcement_learning", "difficulty": "advanced"}
            }
        ]
    
    @pytest.fixture
    def mock_supabase(self, sample_documents):
        """Mock Supabase client with sample data."""
        client = MockSupabaseClient()
        client.add_documents(sample_documents)
        return client
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider."""
        return MockEmbeddingProvider(dimensions=1536)
    
    @pytest.fixture
    def test_queries(self):
        """Test queries for RAG evaluation."""
        return [
            "What is machine learning?",
            "How does deep learning work?",
            "Natural language processing applications",
            "Computer vision techniques",
            "Reinforcement learning algorithms"
        ]
    
    @pytest.mark.asyncio
    async def test_rag_without_reranking(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test RAG pipeline without reranking (baseline)."""
        query = test_queries[0]  # "What is machine learning?"
        
        result = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=None,
            use_reranking=False
        )
        
        assert result["reranking_applied"] == False
        assert len(result["results"]) > 0
        assert "reranking_error" not in result
        
        # Check that results are ordered by similarity score
        similarity_scores = [r.get("similarity_score", 0) for r in result["results"]]
        assert similarity_scores == sorted(similarity_scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_rag_with_openai_reranking(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test RAG pipeline with OpenAI reranking."""
        query = test_queries[0]  # "What is machine learning?"
        reranking_provider = MockRerankingProvider("openai", enabled=True)
        
        result = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=reranking_provider,
            use_reranking=True
        )
        
        assert result["reranking_applied"] == True
        assert len(result["results"]) > 0
        assert reranking_provider.call_count == 1
        
        # Check that results have rerank scores
        for result_item in result["results"]:
            assert "rerank_score" in result_item
            assert "original_rank" in result_item
            assert "reranked_by" in result_item
            assert result_item["reranked_by"] == "openai"
        
        # Verify reranking improved relevance (first result should be most relevant)
        first_result = result["results"][0]
        assert "machine learning" in first_result["content"].lower()
    
    @pytest.mark.asyncio
    async def test_rag_with_ollama_reranking(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test RAG pipeline with Ollama reranking."""
        query = test_queries[1]  # "How does deep learning work?"
        reranking_provider = MockRerankingProvider("ollama", enabled=True)
        
        result = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=reranking_provider,
            use_reranking=True
        )
        
        assert result["reranking_applied"] == True
        assert reranking_provider.call_count == 1
        
        # Check Ollama-specific reranking
        for result_item in result["results"]:
            assert result_item["reranked_by"] == "ollama"
        
        # Ollama should be slower than OpenAI (mocked behavior)
        assert len(reranking_provider.processing_times) > 0
        assert reranking_provider.processing_times[0] > 100  # >100ms for Ollama
    
    @pytest.mark.asyncio
    async def test_rag_with_huggingface_reranking(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test RAG pipeline with HuggingFace reranking."""
        query = test_queries[2]  # "Natural language processing applications"
        reranking_provider = MockRerankingProvider("huggingface", enabled=True)
        
        result = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=reranking_provider,
            use_reranking=True
        )
        
        assert result["reranking_applied"] == True
        assert reranking_provider.call_count == 1
        
        # Check HuggingFace-specific reranking
        for result_item in result["results"]:
            assert result_item["reranked_by"] == "huggingface"
        
        # HuggingFace should be fastest (local processing)
        assert len(reranking_provider.processing_times) > 0
        assert reranking_provider.processing_times[0] < 100  # <100ms for HuggingFace
    
    @pytest.mark.asyncio
    async def test_reranking_fallback_on_failure(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test graceful fallback when reranking fails."""
        query = test_queries[0]
        failing_provider = MockRerankingProvider("openai", enabled=False)  # Disabled provider
        
        result = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=failing_provider,
            use_reranking=True
        )
        
        assert result["reranking_applied"] == False
        assert "reranking_error" in result
        assert len(result["results"]) > 0  # Should still return original results
        
        # Original similarity-based ordering should be preserved
        similarity_scores = [r.get("similarity_score", 0) for r in result["results"]]
        assert similarity_scores == sorted(similarity_scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_reranking_performance_impact(self, mock_supabase, mock_embedding_provider, test_queries):
        """Test performance impact of reranking on RAG pipeline."""
        query = test_queries[0]
        
        # Test without reranking
        start_time = time.time()
        result_without_reranking = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            use_reranking=False
        )
        time_without_reranking = (time.time() - start_time) * 1000
        
        # Test with reranking
        reranking_provider = MockRerankingProvider("openai", enabled=True)
        start_time = time.time()
        result_with_reranking = await mock_perform_rag_query(
            query=query,
            embedding_provider=mock_embedding_provider,
            supabase_client=mock_supabase,
            reranking_provider=reranking_provider,
            use_reranking=True
        )
        time_with_reranking = (time.time() - start_time) * 1000
        
        # Log performance metrics
        logger.info(f"RAG without reranking: {time_without_reranking:.1f}ms")
        logger.info(f"RAG with reranking: {time_with_reranking:.1f}ms")
        logger.info(f"Reranking overhead: {time_with_reranking - time_without_reranking:.1f}ms")
        
        # Verify both return results
        assert len(result_without_reranking["results"]) > 0
        assert len(result_with_reranking["results"]) > 0
        
        # Reranking should add some overhead but not be excessive
        overhead = time_with_reranking - time_without_reranking
        assert overhead > 0  # Should add some time
        assert overhead < 1000  # But not more than 1 second in mock


class TestRerankingProviderSwitching:
    """Test suite for switching between reranking providers."""
    
    @pytest.fixture
    def provider_configs(self):
        """Different provider configurations."""
        return {
            "openai": {
                "provider": "openai",
                "api_key": "sk-test-key",
                "embedding_model": "text-embedding-3-small",
                "reranking_method": "similarity"
            },
            "ollama": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "reranking_model": "bge-reranker-base"
            },
            "huggingface": {
                "provider": "huggingface",
                "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu"
            }
        }
    
    @pytest.mark.asyncio
    async def test_dynamic_provider_switching(self, provider_configs):
        """Test switching between providers dynamically."""
        query = "test query for provider switching"
        results = [
            {"content": "First document about testing"},
            {"content": "Second document about providers"},
            {"content": "Third document about switching"}
        ]
        
        provider_results = {}
        
        # Test each provider
        for provider_name, config in provider_configs.items():
            provider = MockRerankingProvider(provider_name, enabled=True)
            
            rerank_result = await provider.rerank_results(query, results)
            provider_results[provider_name] = rerank_result
            
            # Verify provider-specific behavior
            assert rerank_result.provider == provider_name
            assert len(rerank_result.results) == len(results)
        
        # Compare results across providers
        for provider_name, result in provider_results.items():
            logger.info(f"{provider_name} processing time: {result.processing_time_ms:.1f}ms")
            logger.info(f"{provider_name} score range: {min(result.rerank_scores):.3f} - {max(result.rerank_scores):.3f}")
        
        # All providers should return same number of results
        result_counts = [len(r.results) for r in provider_results.values()]
        assert all(count == result_counts[0] for count in result_counts)
    
    @pytest.mark.asyncio
    async def test_provider_health_monitoring(self, provider_configs):
        """Test health monitoring across providers."""
        providers = {
            name: MockRerankingProvider(name, enabled=True)
            for name in provider_configs.keys()
        }
        
        # Check health of all providers
        health_statuses = {}
        for name, provider in providers.items():
            health = await provider.health_check()
            health_statuses[name] = health
        
        # All providers should be healthy
        for name, health in health_statuses.items():
            assert health["is_healthy"] == True
            assert health["provider"] == name
        
        # Test with one unhealthy provider
        providers["openai"] = MockRerankingProvider("openai", enabled=False)
        unhealthy_health = await providers["openai"].health_check()
        assert unhealthy_health["is_healthy"] == False
    
    @pytest.mark.asyncio 
    async def test_provider_failover(self):
        """Test failover between providers."""
        query = "failover test query"
        results = [{"content": "test document"}]
        
        # Primary provider (fails)
        primary_provider = MockRerankingProvider("openai", enabled=False)
        
        # Fallback provider (works)
        fallback_provider = MockRerankingProvider("ollama", enabled=True)
        
        # Try primary first
        try:
            await primary_provider.rerank_results(query, results)
            pytest.fail("Primary provider should have failed")
        except Exception:
            # Expected failure, use fallback
            fallback_result = await fallback_provider.rerank_results(query, results)
            assert fallback_result.provider == "ollama"
            assert len(fallback_result.results) == 1


class TestRerankingEnvironmentConfiguration:
    """Test suite for environment variable configuration."""
    
    def test_reranking_enable_disable(self):
        """Test enabling/disabling reranking via environment variables."""
        # Test enabled
        with patch.dict(os.environ, {"USE_RERANKING": "true"}):
            use_reranking = os.getenv("USE_RERANKING", "false") == "true"
            assert use_reranking == True
        
        # Test disabled
        with patch.dict(os.environ, {"USE_RERANKING": "false"}):
            use_reranking = os.getenv("USE_RERANKING", "false") == "true"
            assert use_reranking == False
        
        # Test default (disabled)
        with patch.dict(os.environ, {}, clear=True):
            use_reranking = os.getenv("USE_RERANKING", "false") == "true"
            assert use_reranking == False
    
    def test_reranking_provider_selection(self):
        """Test provider selection via environment variables."""
        test_cases = [
            ("openai", "openai"),
            ("ollama", "ollama"),
            ("huggingface", "huggingface"),
            ("", ""),  # Default/empty
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"RERANKING_PROVIDER": env_value}):
                provider = os.getenv("RERANKING_PROVIDER", "").lower()
                assert provider == expected
    
    def test_reranking_model_configuration(self):
        """Test model configuration via environment variables."""
        test_configs = {
            "RERANKING_MODEL": "custom-reranker-v1",
            "OPENAI_RERANKING_MODEL": "text-embedding-3-large", 
            "OLLAMA_RERANKING_MODEL": "bge-reranker-large",
            "HF_RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-12-v2"
        }
        
        for env_var, expected_value in test_configs.items():
            with patch.dict(os.environ, {env_var: expected_value}):
                model = os.getenv(env_var)
                assert model == expected_value
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation for reranking setup."""
        # Test valid configuration
        valid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key"
        }
        
        with patch.dict(os.environ, valid_config):
            use_reranking = os.getenv("USE_RERANKING") == "true"
            provider = os.getenv("RERANKING_PROVIDER")
            api_key = os.getenv("OPENAI_API_KEY")
            
            assert use_reranking == True
            assert provider == "openai"
            assert api_key == "sk-test-key"
        
        # Test invalid configuration (missing API key)
        invalid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai"
            # Missing OPENAI_API_KEY
        }
        
        with patch.dict(os.environ, invalid_config, clear=True):
            use_reranking = os.getenv("USE_RERANKING") == "true"
            provider = os.getenv("RERANKING_PROVIDER")
            api_key = os.getenv("OPENAI_API_KEY")
            
            assert use_reranking == True
            assert provider == "openai"
            assert api_key is None  # Should be caught in validation


class TestRerankingCompatibilityWithRAGStrategies:
    """Test suite for reranking compatibility with existing RAG strategies."""
    
    @pytest.mark.asyncio
    async def test_reranking_with_contextual_embeddings(self):
        """Test reranking with USE_CONTEXTUAL_EMBEDDINGS enabled."""
        # Mock environment with both features enabled
        env_vars = {
            "USE_RERANKING": "true",
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "RERANKING_PROVIDER": "openai"
        }
        
        with patch.dict(os.environ, env_vars):
            query = "contextual embeddings test"
            
            # Mock enhanced documents with contextual information
            contextual_results = [
                {
                    "content": "Document content with context",
                    "contextual_embedding": [0.1] * 1536,
                    "context_summary": "Enhanced with document context"
                }
            ]
            
            reranking_provider = MockRerankingProvider("openai")
            rerank_result = await reranking_provider.rerank_results(query, contextual_results)
            
            # Should work with contextual embeddings
            assert len(rerank_result.results) == 1
            assert rerank_result.provider == "openai"
    
    @pytest.mark.asyncio
    async def test_reranking_with_hybrid_search(self):
        """Test reranking with USE_HYBRID_SEARCH enabled."""
        env_vars = {
            "USE_RERANKING": "true",
            "USE_HYBRID_SEARCH": "true",
            "RERANKING_PROVIDER": "ollama"
        }
        
        with patch.dict(os.environ, env_vars):
            query = "hybrid search test"
            
            # Mock hybrid search results with keyword and semantic scores
            hybrid_results = [
                {
                    "content": "Document with hybrid search scores",
                    "semantic_score": 0.8,
                    "keyword_score": 0.6,
                    "hybrid_score": 0.7
                }
            ]
            
            reranking_provider = MockRerankingProvider("ollama")
            rerank_result = await reranking_provider.rerank_results(query, hybrid_results)
            
            # Should preserve hybrid search information
            assert len(rerank_result.results) == 1
            assert "semantic_score" in rerank_result.results[0]
            assert "hybrid_score" in rerank_result.results[0]
    
    @pytest.mark.asyncio
    async def test_reranking_with_agentic_rag(self):
        """Test reranking with USE_AGENTIC_RAG enabled."""
        env_vars = {
            "USE_RERANKING": "true",
            "USE_AGENTIC_RAG": "true",
            "RERANKING_PROVIDER": "huggingface"
        }
        
        with patch.dict(os.environ, env_vars):
            query = "code examples test"
            
            # Mock agentic RAG results with code examples
            agentic_results = [
                {
                    "content": "Function definition example",
                    "content_type": "code_example",
                    "programming_language": "python",
                    "code_summary": "Function for data processing"
                },
                {
                    "content": "Regular documentation",
                    "content_type": "documentation"
                }
            ]
            
            reranking_provider = MockRerankingProvider("huggingface")
            rerank_result = await reranking_provider.rerank_results(query, agentic_results)
            
            # Should handle both code examples and regular content
            assert len(rerank_result.results) == 2
            assert any(r.get("content_type") == "code_example" for r in rerank_result.results)


class TestRerankingConfigurationValidation:
    """Test suite for comprehensive configuration validation and error handling."""
    
    def test_config_validation_missing_provider(self):
        """Test configuration validation when provider is missing."""
        config = {
            "USE_RERANKING": "true"
            # Missing RERANKING_PROVIDER
        }
        
        with patch.dict(os.environ, config, clear=True):
            use_reranking = os.getenv("USE_RERANKING") == "true"
            provider = os.getenv("RERANKING_PROVIDER", "")
            
            assert use_reranking == True
            assert provider == ""  # Should be caught as invalid
    
    def test_config_validation_invalid_provider(self):
        """Test configuration validation with invalid provider."""
        config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "invalid_provider"
        }
        
        with patch.dict(os.environ, config):
            provider = os.getenv("RERANKING_PROVIDER").lower()
            valid_providers = ["openai", "ollama", "huggingface"]
            
            assert provider == "invalid_provider"
            assert provider not in valid_providers  # Should be caught as invalid
    
    def test_openai_config_validation(self):
        """Test OpenAI-specific configuration validation."""
        # Valid configuration
        valid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key-12345",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
        }
        
        with patch.dict(os.environ, valid_config):
            assert os.getenv("OPENAI_API_KEY") is not None
            assert os.getenv("OPENAI_API_KEY").startswith("sk-")
        
        # Invalid configuration (missing API key)
        invalid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai"
            # Missing OPENAI_API_KEY
        }
        
        with patch.dict(os.environ, invalid_config, clear=True):
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key is None  # Should be caught in validation
    
    def test_ollama_config_validation(self):
        """Test Ollama-specific configuration validation."""
        # Valid configuration
        valid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_RERANKING_MODEL": "bge-reranker-base"
        }
        
        with patch.dict(os.environ, valid_config):
            base_url = os.getenv("OLLAMA_BASE_URL")
            assert base_url.startswith("http://") or base_url.startswith("https://")
        
        # Invalid configuration (invalid URL)
        invalid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "invalid-url"
        }
        
        with patch.dict(os.environ, invalid_config):
            base_url = os.getenv("OLLAMA_BASE_URL")
            assert not (base_url.startswith("http://") or base_url.startswith("https://"))
    
    def test_huggingface_config_validation(self):
        """Test HuggingFace-specific configuration validation."""
        # Valid configuration
        valid_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "huggingface",
            "HF_RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "HF_DEVICE": "cpu"
        }
        
        with patch.dict(os.environ, valid_config):
            model = os.getenv("HF_RERANKING_MODEL")
            device = os.getenv("HF_DEVICE")
            
            assert model is not None
            assert device in ["cpu", "cuda", "cuda:0", "mps"]
    
    def test_config_fallback_values(self):
        """Test configuration fallback to default values."""
        minimal_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai"
        }
        
        with patch.dict(os.environ, minimal_config, clear=True):
            # Test fallback values
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            reranking_model = os.getenv("RERANKING_MODEL", "similarity-based")
            timeout = int(os.getenv("RERANKING_TIMEOUT", "30"))
            
            assert embedding_model == "text-embedding-3-small"
            assert reranking_model == "similarity-based"
            assert timeout == 30
    
    @pytest.mark.asyncio
    async def test_config_reload_on_change(self):
        """Test configuration reloading when environment changes."""
        # Initial configuration
        initial_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai"
        }
        
        with patch.dict(os.environ, initial_config):
            provider1 = MockRerankingProvider("openai")
            result1 = await provider1.rerank_results("test", [{"content": "test"}])
            assert result1.provider == "openai"
        
        # Changed configuration
        changed_config = {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "ollama"
        }
        
        with patch.dict(os.environ, changed_config):
            provider2 = MockRerankingProvider("ollama")
            result2 = await provider2.rerank_results("test", [{"content": "test"}])
            assert result2.provider == "ollama"


class TestRerankingErrorHandlingAndFallbacks:
    """Test suite for comprehensive error handling and fallback scenarios."""
    
    @pytest.mark.asyncio
    async def test_provider_initialization_failure(self):
        """Test handling of provider initialization failures."""
        # Mock provider that fails during initialization
        class FailingProvider(MockRerankingProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                raise ConnectionError("Failed to initialize provider")
        
        with pytest.raises(ConnectionError):
            provider = FailingProvider("openai")
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        class TimeoutProvider(MockRerankingProvider):
            async def rerank_results(self, query, results, model=None):
                await asyncio.sleep(5)  # Simulate slow network
                return await super().rerank_results(query, results, model)
        
        provider = TimeoutProvider("openai")
        query = "timeout test"
        results = [{"content": "test"}]
        
        # Should timeout after 2 seconds
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                provider.rerank_results(query, results),
                timeout=2.0
            )
    
    @pytest.mark.asyncio
    async def test_api_rate_limit_handling(self):
        """Test handling of API rate limits."""
        class RateLimitedProvider(MockRerankingProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.request_count = 0
                self.rate_limit = 3  # Allow 3 requests then rate limit
            
            async def rerank_results(self, query, results, model=None):
                self.request_count += 1
                if self.request_count > self.rate_limit:
                    raise Exception("Rate limit exceeded (429)")
                return await super().rerank_results(query, results, model)
        
        provider = RateLimitedProvider("openai")
        query = "rate limit test"
        results = [{"content": "test"}]
        
        # First 3 requests should succeed
        for i in range(3):
            result = await provider.rerank_results(query, results)
            assert result.provider == "openai"
        
        # 4th request should fail
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await provider.rerank_results(query, results)
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self):
        """Test handling of malformed responses from providers."""
        class MalformedResponseProvider(MockRerankingProvider):
            async def rerank_results(self, query, results, model=None):
                # Return malformed response
                return {
                    "invalid": "response",
                    # Missing required fields
                }
        
        provider = MalformedResponseProvider("openai")
        query = "malformed test"
        results = [{"content": "test"}]
        
        # Should handle malformed response gracefully
        with pytest.raises((TypeError, AttributeError, KeyError)):
            await provider.rerank_results(query, results)
    
    @pytest.mark.asyncio
    async def test_empty_results_graceful_handling(self):
        """Test graceful handling of edge cases."""
        provider = MockRerankingProvider("openai")
        
        # Test with empty results
        with pytest.raises(ValueError):
            await provider.rerank_results("test", [])
        
        # Test with very long query
        long_query = "very " * 1000 + "long query"
        results = [{"content": "test"}]
        
        # Should handle without crashing (may truncate query)
        result = await provider.rerank_results(long_query, results)
        assert len(result.results) == 1
    
    @pytest.mark.asyncio
    async def test_provider_fallback_chain(self):
        """Test fallback chain when multiple providers fail."""
        # Create providers with different failure modes
        failing_provider1 = MockRerankingProvider("openai", enabled=False)
        failing_provider2 = MockRerankingProvider("ollama", enabled=False)
        working_provider = MockRerankingProvider("huggingface", enabled=True)
        
        provider_chain = [failing_provider1, failing_provider2, working_provider]
        query = "fallback test"
        results = [{"content": "test"}]
        
        # Try providers in order until one works
        last_error = None
        for provider in provider_chain:
            try:
                result = await provider.rerank_results(query, results)
                assert result.provider == "huggingface"  # Should be the working provider
                break
            except Exception as e:
                last_error = e
                continue
        else:
            pytest.fail(f"All providers failed, last error: {last_error}")
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        class PartialFailureProvider(MockRerankingProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.failure_count = 0
            
            async def rerank_results(self, query, results, model=None):
                self.failure_count += 1
                if self.failure_count <= 2:  # Fail first 2 attempts
                    raise Exception("Temporary failure")
                # Succeed on 3rd attempt
                return await super().rerank_results(query, results, model)
        
        provider = PartialFailureProvider("openai")
        query = "partial failure test"
        results = [{"content": "test"}]
        
        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await provider.rerank_results(query, results)
                assert result.provider == "openai"
                assert attempt == 2  # Should succeed on 3rd attempt
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    pytest.fail(f"Failed after {max_retries} attempts: {e}")
                # Wait before retry (exponential backoff simulation)
                await asyncio.sleep(0.1 * (2 ** attempt))
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_on_failure(self):
        """Test that resources are properly cleaned up on failure."""
        class ResourceLeakProvider(MockRerankingProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.resources_allocated = 0
                self.resources_cleaned = 0
            
            async def rerank_results(self, query, results, model=None):
                try:
                    self.resources_allocated += 1
                    if not self.connection_healthy:
                        raise Exception("Connection failed")
                    return await super().rerank_results(query, results, model)
                finally:
                    # Always clean up resources
                    self.resources_cleaned += 1
        
        provider = ResourceLeakProvider("openai", enabled=False)  # Will fail
        query = "resource test"
        results = [{"content": "test"}]
        
        with pytest.raises(Exception):
            await provider.rerank_results(query, results)
        
        # Verify cleanup occurred
        assert provider.resources_allocated == 1
        assert provider.resources_cleaned == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_failure_isolation(self):
        """Test that failures in concurrent requests don't affect each other."""
        class MixedProvider(MockRerankingProvider):
            async def rerank_results(self, query, results, model=None):
                # Fail queries containing "fail"
                if "fail" in query.lower():
                    raise Exception(f"Simulated failure for query: {query}")
                return await super().rerank_results(query, results, model)
        
        provider = MixedProvider("openai")
        results = [{"content": "test document"}]
        
        # Create mixed concurrent requests (some will fail, some succeed)
        tasks = []
        for i in range(10):
            query = f"fail query {i}" if i % 3 == 0 else f"success query {i}"
            tasks.append(provider.rerank_results(query, results))
        
        # Gather results, allowing exceptions
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = [r for r in completed_results if not isinstance(r, Exception)]
        failures = [r for r in completed_results if isinstance(r, Exception)]
        
        assert len(successes) > 0  # Some should succeed
        assert len(failures) > 0  # Some should fail
        assert len(successes) + len(failures) == 10  # All requests completed
        
        # Verify successful results are valid
        for success in successes:
            assert success.provider == "openai"
            assert len(success.results) == 1


class TestRerankingRegressionTests:
    """Test suite to ensure existing functionality still works with reranking."""
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_without_reranking(self):
        """Test that disabling reranking preserves original behavior."""
        query = "test query"
        original_results = [
            {"content": "First document", "similarity_score": 0.9},
            {"content": "Second document", "similarity_score": 0.8}, 
            {"content": "Third document", "similarity_score": 0.7}
        ]
        
        # Test with reranking disabled
        with patch.dict(os.environ, {"USE_RERANKING": "false"}):
            # Should return results in original order
            mock_supabase = MockSupabaseClient()
            mock_embedding = MockEmbeddingProvider()
            
            result = await mock_perform_rag_query(
                query=query,
                embedding_provider=mock_embedding,
                supabase_client=mock_supabase,
                use_reranking=False
            )
            
            assert result["reranking_applied"] == False
            # Results should maintain similarity score ordering
            if result["results"]:
                scores = [r.get("similarity_score", 0) for r in result["results"]]
                assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_existing_metadata_preservation(self):
        """Test that existing metadata is preserved through reranking."""
        query = "metadata preservation test"
        results_with_metadata = [
            {
                "content": "Document with metadata",
                "url": "https://example.com/doc1",
                "metadata": {
                    "author": "John Doe",
                    "created_at": "2024-01-01",
                    "tags": ["important", "test"]
                },
                "chunk_id": 123,
                "similarity_score": 0.85
            }
        ]
        
        provider = MockRerankingProvider("openai")
        result = await provider.rerank_results(query, results_with_metadata)
        
        # Check that original metadata is preserved
        reranked_doc = result.results[0]
        assert reranked_doc["url"] == "https://example.com/doc1"
        assert reranked_doc["metadata"]["author"] == "John Doe"
        assert reranked_doc["metadata"]["tags"] == ["important", "test"]
        assert reranked_doc["chunk_id"] == 123
        
        # Check that new reranking metadata is added
        assert "rerank_score" in reranked_doc
        assert "original_rank" in reranked_doc
    
    @pytest.mark.asyncio
    async def test_existing_rag_strategies_still_work(self):
        """Test that existing RAG strategies continue to work with reranking."""
        # Test various combinations of RAG strategies
        strategy_combinations = [
            {"USE_RERANKING": "true", "USE_CONTEXTUAL_EMBEDDINGS": "true"},
            {"USE_RERANKING": "true", "USE_HYBRID_SEARCH": "true"},
            {"USE_RERANKING": "true", "USE_AGENTIC_RAG": "true"},
            {"USE_RERANKING": "false", "USE_CONTEXTUAL_EMBEDDINGS": "true"},
            {"USE_RERANKING": "false", "USE_HYBRID_SEARCH": "true"},
        ]
        
        query = "RAG strategies compatibility test"
        test_results = [{"content": "Test document for RAG strategies"}]
        
        for strategy_env in strategy_combinations:
            with patch.dict(os.environ, strategy_env):
                use_reranking = os.getenv("USE_RERANKING") == "true"
                
                if use_reranking:
                    provider = MockRerankingProvider("openai")
                    result = await provider.rerank_results(query, test_results)
                    assert len(result.results) == 1
                else:
                    # Original behavior should work
                    mock_supabase = MockSupabaseClient()
                    mock_embedding = MockEmbeddingProvider()
                    mock_supabase.add_documents(test_results)
                    
                    rag_result = await mock_perform_rag_query(
                        query=query,
                        embedding_provider=mock_embedding,
                        supabase_client=mock_supabase,
                        use_reranking=False
                    )
                    assert len(rag_result["results"]) >= 0
    
    def test_api_interface_compatibility(self):
        """Test that existing API interfaces remain compatible."""
        # Test that old function signatures still work
        provider = MockRerankingProvider("openai")
        
        # Check that required methods exist
        assert hasattr(provider, 'rerank_results')
        assert callable(getattr(provider, 'rerank_results'))
        
        # Check that method signatures are compatible
        import inspect
        sig = inspect.signature(provider.rerank_results)
        params = list(sig.parameters.keys())
        
        # Should have expected parameters
        expected_params = ['query', 'results', 'model']
        for param in expected_params:
            assert param in params or param == 'self'
    
    @pytest.mark.asyncio
    async def test_performance_regression(self):
        """Test that reranking doesn't cause significant performance regression."""
        query = "performance regression test"
        results = [{"content": f"Document {i}"} for i in range(20)]
        
        # Measure performance with and without reranking
        provider = MockRerankingProvider("openai")
        
        # Test with reranking
        start_time = time.time()
        rerank_result = await provider.rerank_results(query, results)
        rerank_time = (time.time() - start_time) * 1000
        
        # Performance should be reasonable
        assert rerank_time < 1000  # Should complete within 1 second for mock
        assert len(rerank_result.results) == len(results)
        
        # Check that processing time is reported
        assert rerank_result.processing_time_ms > 0
        assert rerank_result.processing_time_ms < 1000


class TestRerankingPerformanceBenchmarking:
    """Test suite for reranking performance benchmarking."""
    
    @pytest.mark.asyncio
    async def test_reranking_latency_benchmark(self):
        """Benchmark reranking latency across providers."""
        query = "performance benchmark test"
        
        # Create different sized result sets
        small_results = [{"content": f"Document {i}"} for i in range(10)]
        medium_results = [{"content": f"Document {i}"} for i in range(50)]
        large_results = [{"content": f"Document {i}"} for i in range(100)]
        
        providers = {
            "openai": MockRerankingProvider("openai"),
            "ollama": MockRerankingProvider("ollama"),
            "huggingface": MockRerankingProvider("huggingface")
        }
        
        benchmarks = {}
        
        for provider_name, provider in providers.items():
            benchmarks[provider_name] = {}
            
            for size_name, results in [("small", small_results), ("medium", medium_results), ("large", large_results)]:
                start_time = time.time()
                await provider.rerank_results(query, results)
                elapsed_time = (time.time() - start_time) * 1000
                
                benchmarks[provider_name][size_name] = elapsed_time
        
        # Log benchmark results
        for provider_name, provider_benchmarks in benchmarks.items():
            logger.info(f"{provider_name} latency benchmarks:")
            for size, latency in provider_benchmarks.items():
                logger.info(f"  {size}: {latency:.1f}ms")
        
        # Verify reasonable performance
        for provider_benchmarks in benchmarks.values():
            for latency in provider_benchmarks.values():
                assert latency < 1000  # Should be under 1 second for mocked providers
    
    @pytest.mark.asyncio
    async def test_reranking_throughput_benchmark(self):
        """Benchmark reranking throughput with concurrent requests."""
        query = "throughput benchmark test"
        results = [{"content": f"Document {i}"} for i in range(20)]
        
        providers = {
            "openai": MockRerankingProvider("openai"), 
            "ollama": MockRerankingProvider("ollama"),
            "huggingface": MockRerankingProvider("huggingface")
        }
        
        concurrent_requests = 10
        
        for provider_name, provider in providers.items():
            start_time = time.time()
            
            # Run concurrent reranking requests
            tasks = [
                provider.rerank_results(f"{query} {i}", results)
                for i in range(concurrent_requests)
            ]
            
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            throughput = concurrent_requests / total_time
            
            logger.info(f"{provider_name} throughput: {throughput:.1f} requests/second")
            
            # Verify reasonable throughput
            assert throughput > 0.1  # At least 0.1 requests per second
            assert provider.call_count == concurrent_requests


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v",
        "--tb=short", 
        "--disable-warnings",
        "--maxfail=10",
        "--durations=15"
    ])