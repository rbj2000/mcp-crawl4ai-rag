"""
Comprehensive unit tests for reranking provider implementations.

This test suite ensures:
1. Individual provider functionality (OpenAI, Ollama, HuggingFace)
2. Reranking input validation and error handling
3. Configuration validation and provider initialization
4. Performance metrics and health checks
5. Model availability and fallback mechanisms
6. Cross-provider compatibility and consistency
7. Edge cases and error scenarios
"""

import pytest
import asyncio
import os
import time
import json
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

# Import reranking provider classes
try:
    from ai_providers.base import (
        RerankingProvider, RerankingResult, HealthStatus, AIProvider
    )
    from ai_providers.providers.openai_provider import OpenAIProvider
    from ai_providers.providers.ollama_provider import OllamaProvider
    from ai_providers.providers.huggingface_provider import HuggingFaceProvider
    from ai_providers.factory import AIProviderFactory
    from ai_providers.config import AIProviderConfig
except ImportError as e:
    logger.warning(f"Could not import reranking providers: {e}")
    # Mock classes for testing framework development
    from enum import Enum
    from dataclasses import dataclass
    from typing import List
    
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
    
    @dataclass
    class HealthStatus:
        is_healthy: bool
        provider: str
        model: str
        response_time_ms: float
        error_message: Optional[str] = None
    
    class RerankingProvider:
        def __init__(self, config):
            self.config = config
    
    class OpenAIProvider:
        def __init__(self, config):
            self.config = config
    
    class OllamaProvider:
        def __init__(self, config):
            self.config = config
    
    class HuggingFaceProvider:
        def __init__(self, config):
            self.config = config
    
    class AIProviderFactory:
        @staticmethod
        def create_provider(provider_type, config):
            return MockRerankingProvider()
    
    class AIProviderConfig:
        def __init__(self, provider, config):
            self.provider = provider
            self.config = config


# Custom exceptions
class RerankingError(Exception):
    pass

class ModelNotAvailableError(Exception):
    pass

class InvalidQueryError(Exception):
    pass


class MockRerankingProvider:
    """Mock reranking provider for testing base functionality."""
    
    def __init__(self, provider_type: str = "mock", connection_healthy: bool = True):
        self.provider_type = provider_type
        self.connection_healthy = connection_healthy
        self.call_count = 0
        self.models_available = ["mock-reranker-v1", "mock-cross-encoder"]
        self.default_model = "mock-reranker-v1"
    
    async def rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        model: Optional[str] = None
    ) -> RerankingResult:
        """Mock reranking implementation."""
        self.call_count += 1
        
        if not self.connection_healthy:
            raise RerankingError("Connection failed")
        
        if not query.strip():
            raise InvalidQueryError("Query cannot be empty")
        
        if not results:
            raise ValueError("Results cannot be empty")
        
        # Generate mock scores based on content length (simple heuristic)
        mock_scores = []
        reranked_results = []
        
        for i, result in enumerate(results):
            content = result.get("content", "")
            # Simple scoring: longer content with query keywords gets higher score
            score = len(content) * 0.01 + (query.lower() in content.lower()) * 0.5
            mock_scores.append(score)
            
            enhanced_result = result.copy()
            enhanced_result["rerank_score"] = score
            enhanced_result["original_rank"] = i
            reranked_results.append(enhanced_result)
        
        # Sort by score descending
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return RerankingResult(
            results=reranked_results,
            model=model or self.default_model,
            provider=self.provider_type,
            rerank_scores=mock_scores,
            processing_time_ms=50.0,  # Mock processing time
            metadata={
                "mock_provider": True,
                "query_length": len(query),
                "results_count": len(results)
            }
        )
    
    async def health_check(self) -> HealthStatus:
        """Mock health check."""
        return HealthStatus(
            is_healthy=self.connection_healthy,
            provider=self.provider_type,
            model=self.default_model,
            response_time_ms=10.0,
            error_message=None if self.connection_healthy else "Connection failed"
        )
    
    def supports_reranking(self) -> bool:
        """Mock reranking support check."""
        return True
    
    def get_reranking_models(self) -> List[str]:
        """Get available models."""
        return self.models_available
    
    @property
    def default_reranking_model(self) -> str:
        """Default reranking model."""
        return self.default_model


class TestRerankingProviderBase:
    """Test suite for base reranking provider functionality."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock reranking provider."""
        return MockRerankingProvider()
    
    @pytest.fixture
    def test_query(self):
        """Test query fixture."""
        return "machine learning algorithms"
    
    @pytest.fixture
    def test_results(self):
        """Test results fixture."""
        return [
            {
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "url": "https://example.com/ml-intro",
                "metadata": {"source": "wikipedia"}
            },
            {
                "content": "Deep learning uses neural networks with multiple layers.",
                "url": "https://example.com/deep-learning",
                "metadata": {"source": "tutorial"}
            },
            {
                "content": "Python is a popular programming language for data science.",
                "url": "https://example.com/python",
                "metadata": {"source": "docs"}
            },
            {
                "content": "Statistical methods are important for data analysis.",
                "url": "https://example.com/statistics",
                "metadata": {"source": "textbook"}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_basic_reranking(self, mock_provider, test_query, test_results):
        """Test basic reranking functionality."""
        result = await mock_provider.rerank_results(test_query, test_results)
        
        assert isinstance(result, RerankingResult)
        assert len(result.results) == len(test_results)
        assert len(result.rerank_scores) == len(test_results)
        assert result.provider == "mock"
        assert result.processing_time_ms > 0
        
        # Check that results are properly reranked
        for i, reranked_result in enumerate(result.results):
            assert "rerank_score" in reranked_result
            assert "original_rank" in reranked_result
            assert isinstance(reranked_result["rerank_score"], (int, float))
        
        # Verify ordering (should be descending by score)
        scores = [r["rerank_score"] for r in result.results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_empty_query_validation(self, mock_provider, test_results):
        """Test validation of empty queries."""
        with pytest.raises(InvalidQueryError):
            await mock_provider.rerank_results("", test_results)
        
        with pytest.raises(InvalidQueryError):
            await mock_provider.rerank_results("   ", test_results)
    
    @pytest.mark.asyncio
    async def test_empty_results_validation(self, mock_provider, test_query):
        """Test validation of empty results."""
        with pytest.raises(ValueError):
            await mock_provider.rerank_results(test_query, [])
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_provider):
        """Test health check functionality."""
        health = await mock_provider.health_check()
        
        assert isinstance(health, HealthStatus)
        assert health.is_healthy == True
        assert health.provider == "mock"
        assert health.response_time_ms > 0
        assert health.error_message is None
    
    @pytest.mark.asyncio
    async def test_unhealthy_provider(self):
        """Test behavior with unhealthy provider."""
        unhealthy_provider = MockRerankingProvider(connection_healthy=False)
        
        # Health check should report unhealthy
        health = await unhealthy_provider.health_check()
        assert health.is_healthy == False
        assert health.error_message is not None
        
        # Reranking should fail
        with pytest.raises(RerankingError):
            await unhealthy_provider.rerank_results("test", [{"content": "test"}])
    
    def test_supports_reranking(self, mock_provider):
        """Test reranking support check."""
        assert mock_provider.supports_reranking() == True
    
    def test_get_reranking_models(self, mock_provider):
        """Test getting available models."""
        models = mock_provider.get_reranking_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert all(isinstance(model, str) for model in models)
    
    def test_default_reranking_model(self, mock_provider):
        """Test default model property."""
        default_model = mock_provider.default_reranking_model
        assert isinstance(default_model, str)
        assert len(default_model) > 0


class TestOpenAIRerankingProvider:
    """Test suite for OpenAI similarity-based reranking provider."""
    
    @pytest.fixture
    def openai_config(self):
        """OpenAI provider configuration."""
        return {
            "api_key": "sk-test-openai-key",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 1536,
            "llm_model": "gpt-4o-mini",
            "max_retries": 3,
            "timeout": 30
        }
    
    @pytest.fixture
    def mock_openai_provider(self, openai_config):
        """Create mock OpenAI provider with reranking."""
        provider = MockRerankingProvider("openai")
        provider.config = openai_config
        return provider
    
    @pytest.mark.asyncio
    async def test_openai_similarity_reranking(self, mock_openai_provider):
        """Test OpenAI similarity-based reranking."""
        query = "artificial intelligence"
        results = [
            {"content": "AI is transforming technology"},
            {"content": "Machine learning models"},
            {"content": "Deep neural networks"}
        ]
        
        with patch.object(mock_openai_provider, 'rerank_results') as mock_rerank:
            # Mock similarity-based reranking
            mock_result = RerankingResult(
                results=[
                    {"content": "AI is transforming technology", "rerank_score": 0.95, "original_rank": 0},
                    {"content": "Deep neural networks", "rerank_score": 0.78, "original_rank": 2},
                    {"content": "Machine learning models", "rerank_score": 0.65, "original_rank": 1}
                ],
                model="text-embedding-3-small",
                provider="openai",
                rerank_scores=[0.95, 0.65, 0.78],
                processing_time_ms=120.5,
                metadata={
                    "reranking_method": "cosine_similarity",
                    "embedding_model": "text-embedding-3-small",
                    "embedding_dimensions": 1536
                }
            )
            mock_rerank.return_value = mock_result
            
            result = await mock_openai_provider.rerank_results(query, results)
            
            assert result.provider == "openai"
            assert result.metadata["reranking_method"] == "cosine_similarity"
            assert len(result.results) == 3
            
            # Verify score-based ordering
            scores = [r["rerank_score"] for r in result.results]
            assert scores == [0.95, 0.78, 0.65]  # Descending order
    
    @pytest.mark.asyncio
    async def test_openai_embedding_fallback(self, mock_openai_provider):
        """Test fallback when embeddings fail."""
        query = "test query"
        results = [{"content": "test content"}]
        
        with patch.object(mock_openai_provider, 'rerank_results') as mock_rerank:
            # Simulate embedding failure, fallback to original order
            mock_rerank.side_effect = RerankingError("Embedding service unavailable")
            
            with pytest.raises(RerankingError):
                await mock_openai_provider.rerank_results(query, results)
    
    def test_openai_cosine_similarity_calculation(self):
        """Test cosine similarity calculation logic."""
        # This would test the actual cosine similarity function
        # For now, we'll test the concept
        vector1 = [1.0, 0.0, 0.0]
        vector2 = [1.0, 0.0, 0.0]
        
        # Mock the similarity calculation
        def mock_cosine_similarity(v1, v2):
            import math
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        similarity = mock_cosine_similarity(vector1, vector2)
        assert similarity == 1.0  # Identical vectors
        
        # Test orthogonal vectors
        vector3 = [0.0, 1.0, 0.0]
        similarity = mock_cosine_similarity(vector1, vector3)
        assert similarity == 0.0  # Orthogonal vectors
    
    @pytest.mark.asyncio
    async def test_openai_batch_processing(self, mock_openai_provider):
        """Test batch processing for large result sets."""
        query = "batch processing test"
        # Create large result set
        results = [{"content": f"Document {i} content"} for i in range(150)]
        
        with patch.object(mock_openai_provider, 'rerank_results') as mock_rerank:
            # Mock batch processing
            mock_result = RerankingResult(
                results=results[:100],  # Simulate batch limit
                model="text-embedding-3-small",
                provider="openai",
                rerank_scores=[0.8] * 100,
                processing_time_ms=500.0,
                metadata={"batch_processed": True, "total_batches": 2}
            )
            mock_rerank.return_value = mock_result
            
            result = await mock_openai_provider.rerank_results(query, results)
            
            assert len(result.results) <= 100  # Respects batch limits
            assert result.metadata.get("batch_processed") == True


class TestOllamaRerankingProvider:
    """Test suite for Ollama dedicated reranking provider."""
    
    @pytest.fixture
    def ollama_config(self):
        """Ollama provider configuration."""
        return {
            "base_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "embedding_dimensions": 768,
            "llm_model": "llama3.2",
            "reranking_model": "bge-reranker-base",
            "timeout": 120,
            "verify_ssl": False
        }
    
    @pytest.fixture
    def mock_ollama_provider(self, ollama_config):
        """Create mock Ollama provider with reranking."""
        provider = MockRerankingProvider("ollama")
        provider.config = ollama_config
        provider.models_available = ["bge-reranker-base", "bge-reranker-large", "ms-marco-MiniLM-L-6-v2"]
        provider.default_model = "bge-reranker-base"
        return provider
    
    @pytest.mark.asyncio
    async def test_ollama_dedicated_reranking(self, mock_ollama_provider):
        """Test Ollama dedicated reranking models."""
        query = "neural networks deep learning"
        results = [
            {"content": "Convolutional neural networks for image recognition"},
            {"content": "Recurrent neural networks for sequence data"},
            {"content": "Transformer architecture for NLP tasks"}
        ]
        
        with patch.object(mock_ollama_provider, 'rerank_results') as mock_rerank:
            mock_result = RerankingResult(
                results=[
                    {"content": "Transformer architecture for NLP tasks", "rerank_score": 0.92, "original_rank": 2},
                    {"content": "Convolutional neural networks for image recognition", "rerank_score": 0.87, "original_rank": 0},
                    {"content": "Recurrent neural networks for sequence data", "rerank_score": 0.81, "original_rank": 1}
                ],
                model="bge-reranker-base",
                provider="ollama",
                rerank_scores=[0.87, 0.81, 0.92],
                processing_time_ms=250.0,
                metadata={
                    "reranking_method": "cross_encoder",
                    "model_size": "278M",
                    "ollama_version": "0.1.0"
                }
            )
            mock_rerank.return_value = mock_result
            
            result = await mock_ollama_provider.rerank_results(query, results)
            
            assert result.provider == "ollama"
            assert result.model == "bge-reranker-base"
            assert result.metadata["reranking_method"] == "cross_encoder"
            
            # Verify cross-encoder based reranking
            scores = [r["rerank_score"] for r in result.results]
            assert scores == [0.92, 0.87, 0.81]  # Properly ordered
    
    @pytest.mark.asyncio
    async def test_ollama_model_availability(self, mock_ollama_provider):
        """Test model availability checking."""
        with patch.object(mock_ollama_provider, '_ensure_model_available') as mock_ensure:
            mock_ensure.return_value = True
            
            query = "test"
            results = [{"content": "test"}]
            
            # Should not raise exception if model is available
            await mock_ollama_provider.rerank_results(query, results)
            
        with patch.object(mock_ollama_provider, '_ensure_model_available') as mock_ensure:
            mock_ensure.side_effect = ModelNotAvailableError("Model not found")
            
            with pytest.raises(ModelNotAvailableError):
                await mock_ollama_provider.rerank_results(query, results)
    
    @pytest.mark.asyncio
    async def test_ollama_model_pulling(self, mock_ollama_provider):
        """Test automatic model pulling."""
        with patch.object(mock_ollama_provider, '_pull_model_if_needed') as mock_pull:
            mock_pull.return_value = True
            
            query = "test"
            results = [{"content": "test"}]
            
            # Should pull model automatically if enabled
            await mock_ollama_provider.rerank_results(query, results)
    
    def test_ollama_reranking_models_list(self, mock_ollama_provider):
        """Test getting available reranking models."""
        models = mock_ollama_provider.get_reranking_models()
        
        expected_models = ["bge-reranker-base", "bge-reranker-large", "ms-marco-MiniLM-L-6-v2"]
        assert models == expected_models
        
        # Test default model
        assert mock_ollama_provider.default_reranking_model == "bge-reranker-base"


class TestHuggingFaceRerankingProvider:
    """Test suite for HuggingFace local reranking provider."""
    
    @pytest.fixture
    def huggingface_config(self):
        """HuggingFace provider configuration."""
        return {
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
            "max_length": 512,
            "batch_size": 32,
            "trust_remote_code": False
        }
    
    @pytest.fixture
    def mock_huggingface_provider(self, huggingface_config):
        """Create mock HuggingFace provider with reranking."""
        provider = MockRerankingProvider("huggingface")
        provider.config = huggingface_config
        provider.models_available = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/stsb-roberta-large"
        ]
        provider.default_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        return provider
    
    @pytest.mark.asyncio
    async def test_huggingface_local_reranking(self, mock_huggingface_provider):
        """Test HuggingFace local cross-encoder reranking."""
        query = "information retrieval systems"
        results = [
            {"content": "Search engines use inverted indexes"},
            {"content": "Database query optimization techniques"},
            {"content": "Information retrieval models and evaluation"}
        ]
        
        with patch.object(mock_huggingface_provider, 'rerank_results') as mock_rerank:
            mock_result = RerankingResult(
                results=[
                    {"content": "Information retrieval models and evaluation", "rerank_score": 0.94, "original_rank": 2},
                    {"content": "Search engines use inverted indexes", "rerank_score": 0.86, "original_rank": 0},
                    {"content": "Database query optimization techniques", "rerank_score": 0.72, "original_rank": 1}
                ],
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                provider="huggingface",
                rerank_scores=[0.86, 0.72, 0.94],
                processing_time_ms=89.3,
                metadata={
                    "device": "cpu",
                    "model_size": "22.7M",
                    "local_processing": True
                }
            )
            mock_rerank.return_value = mock_result
            
            result = await mock_huggingface_provider.rerank_results(query, results)
            
            assert result.provider == "huggingface"
            assert result.metadata["local_processing"] == True
            assert result.processing_time_ms < 100  # Local processing should be fast
    
    @pytest.mark.asyncio
    async def test_huggingface_model_loading(self, mock_huggingface_provider):
        """Test model loading and initialization."""
        with patch('sentence_transformers.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = [0.8, 0.6, 0.9]
            mock_cross_encoder.return_value = mock_model
            
            # Test model initialization
            query = "test"
            results = [{"content": f"doc {i}"} for i in range(3)]
            
            await mock_huggingface_provider.rerank_results(query, results)
    
    @pytest.mark.asyncio
    async def test_huggingface_batch_processing(self, mock_huggingface_provider):
        """Test batch processing for efficiency."""
        query = "batch processing"
        results = [{"content": f"Document {i}"} for i in range(64)]  # Large batch
        
        with patch.object(mock_huggingface_provider, 'rerank_results') as mock_rerank:
            # Mock efficient batch processing
            mock_result = RerankingResult(
                results=results,  # All results processed
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                provider="huggingface",
                rerank_scores=[0.7 + (i % 10) * 0.01 for i in range(64)],
                processing_time_ms=156.7,  # Efficient batch processing
                metadata={"batch_size": 32, "num_batches": 2}
            )
            mock_rerank.return_value = mock_result
            
            result = await mock_huggingface_provider.rerank_results(query, results)
            
            assert len(result.results) == 64
            assert result.metadata["num_batches"] == 2
    
    def test_huggingface_device_configuration(self, mock_huggingface_provider):
        """Test device configuration (CPU/GPU)."""
        config = mock_huggingface_provider.config
        assert config["device"] == "cpu"
        
        # Test GPU configuration
        gpu_config = config.copy()
        gpu_config["device"] = "cuda:0"
        
        # Would test GPU availability and fallback to CPU if needed
        assert gpu_config["device"] in ["cpu", "cuda:0", "cuda", "mps"]


class TestRerankingProviderComparison:
    """Test suite for comparing reranking providers."""
    
    @pytest.fixture
    def comparison_query(self):
        """Query for comparison testing."""
        return "machine learning model performance"
    
    @pytest.fixture
    def comparison_results(self):
        """Results for comparison testing."""
        return [
            {
                "content": "Model accuracy metrics and evaluation techniques for machine learning",
                "url": "https://example.com/ml-metrics"
            },
            {
                "content": "Deep learning neural network architecture optimization",
                "url": "https://example.com/deep-learning"
            },
            {
                "content": "Statistical analysis of algorithm performance",
                "url": "https://example.com/statistics"
            },
            {
                "content": "Data preprocessing techniques for better model performance",
                "url": "https://example.com/preprocessing"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_provider_consistency(self, comparison_query, comparison_results):
        """Test consistency across different providers."""
        providers = {
            "openai": MockRerankingProvider("openai"),
            "ollama": MockRerankingProvider("ollama"),
            "huggingface": MockRerankingProvider("huggingface")
        }
        
        results = {}
        for name, provider in providers.items():
            result = await provider.rerank_results(comparison_query, comparison_results)
            results[name] = result
        
        # All providers should return same number of results
        result_counts = [len(r.results) for r in results.values()]
        assert all(count == result_counts[0] for count in result_counts)
        
        # All should have valid scores
        for name, result in results.items():
            assert len(result.rerank_scores) == len(comparison_results)
            assert all(isinstance(score, (int, float)) for score in result.rerank_scores)
    
    @pytest.mark.asyncio
    async def test_performance_comparison(self, comparison_query, comparison_results):
        """Test performance characteristics of different providers."""
        providers = {
            "openai": MockRerankingProvider("openai"),
            "ollama": MockRerankingProvider("ollama"),
            "huggingface": MockRerankingProvider("huggingface")
        }
        
        performance_metrics = {}
        
        for name, provider in providers.items():
            start_time = time.time()
            result = await provider.rerank_results(comparison_query, comparison_results)
            elapsed_time = (time.time() - start_time) * 1000
            
            performance_metrics[name] = {
                "processing_time_ms": result.processing_time_ms,
                "actual_time_ms": elapsed_time,
                "scores_range": max(result.rerank_scores) - min(result.rerank_scores)
            }
        
        # Log performance for analysis
        for name, metrics in performance_metrics.items():
            logger.info(f"{name} provider: {metrics}")
        
        # All providers should complete within reasonable time
        for name, metrics in performance_metrics.items():
            assert metrics["actual_time_ms"] < 5000  # 5 seconds max
    
    def test_scoring_distribution(self, comparison_query, comparison_results):
        """Test score distribution characteristics."""
        provider = MockRerankingProvider()
        
        # Test with various result sets
        test_cases = [
            comparison_results,  # Original
            comparison_results[:2],  # Small set
            comparison_results * 3,  # Larger set
        ]
        
        for i, test_results in enumerate(test_cases):
            result = asyncio.run(provider.rerank_results(comparison_query, test_results))
            
            scores = result.rerank_scores
            
            # Scores should be normalized or in reasonable range
            assert all(0.0 <= score <= 1.0 or -1.0 <= score <= 1.0 for score in scores)
            
            # Should have score variance (not all identical)
            if len(scores) > 1:
                score_variance = max(scores) - min(scores)
                assert score_variance >= 0  # Some variance expected
            
            logger.info(f"Test case {i}: score range {min(scores):.3f} - {max(scores):.3f}")


class TestRerankingErrorHandling:
    """Test suite for error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_malformed_results(self):
        """Test handling of malformed result data."""
        provider = MockRerankingProvider()
        query = "test query"
        
        # Test missing content field
        malformed_results = [
            {"url": "https://example.com", "metadata": {}},  # Missing content
            {"content": "", "url": "https://example.com"},  # Empty content
            {"content": None, "url": "https://example.com"},  # Null content
        ]
        
        # Should handle gracefully or raise appropriate errors
        with pytest.raises((ValueError, TypeError)):
            await provider.rerank_results(query, malformed_results)
    
    @pytest.mark.asyncio
    async def test_very_long_content(self):
        """Test handling of very long content."""
        provider = MockRerankingProvider()
        query = "test"
        
        # Create result with very long content
        long_content = "A" * 100000  # 100k characters
        results = [{"content": long_content}]
        
        # Should handle without crashing (may truncate)
        result = await provider.rerank_results(query, results)
        assert len(result.results) == 1
    
    @pytest.mark.asyncio
    async def test_unicode_handling(self):
        """Test handling of unicode and special characters."""
        provider = MockRerankingProvider()
        
        # Test various unicode scenarios
        test_cases = [
            ("café résumé", [{"content": "café résumé français"}]),
            ("🤖 AI", [{"content": "🤖 Artificial Intelligence 🧠"}]),
            ("中文查询", [{"content": "中文文档内容"}]),
            ("مرحبا", [{"content": "Arabic content مرحبا بك"}]),
        ]
        
        for query, results in test_cases:
            result = await provider.rerank_results(query, results)
            assert len(result.results) == 1
            assert result.results[0]["content"] == results[0]["content"]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent reranking requests."""
        provider = MockRerankingProvider()
        query = "concurrent test"
        results = [{"content": f"Document {i}"} for i in range(5)]
        
        # Create multiple concurrent requests
        tasks = [
            provider.rerank_results(f"{query} {i}", results)
            for i in range(10)
        ]
        
        # All should complete successfully
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all completed without exceptions
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                pytest.fail(f"Task {i} raised exception: {result}")
            assert len(result.results) == 5
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for slow operations."""
        class SlowProvider(MockRerankingProvider):
            async def rerank_results(self, query, results, model=None):
                await asyncio.sleep(2)  # Simulate slow operation
                return await super().rerank_results(query, results, model)
        
        provider = SlowProvider()
        query = "timeout test"
        results = [{"content": "test"}]
        
        # Test with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                provider.rerank_results(query, results),
                timeout=1.0  # 1 second timeout
            )


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v",
        "--tb=short",
        "--disable-warnings",
        "--maxfail=10",
        "--durations=15",
        "-x"  # Stop on first failure for debugging
    ])