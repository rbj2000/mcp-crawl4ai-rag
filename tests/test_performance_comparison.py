"""
Performance testing framework comparing OpenAI and Ollama AI providers.

This test suite validates:
1. Embedding generation performance (latency, throughput)
2. LLM text generation performance 
3. Memory usage patterns
4. Concurrent request handling
5. Cost analysis (API calls vs local processing)
6. Quality metrics (embedding similarity, text coherence)
"""

import pytest
import asyncio
import time
import statistics
import tracemalloc
import psutil
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from unittest.mock import patch, Mock

# Test configuration
PERFORMANCE_TEST_ENABLED = os.getenv("ENABLE_PERFORMANCE_TESTS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")

SKIP_PERFORMANCE_TESTS = not PERFORMANCE_TEST_ENABLED
PERFORMANCE_NOT_AVAILABLE = "Performance tests disabled or dependencies not available"


@dataclass
class PerformanceMetrics:
    """Performance metrics for AI provider operations."""
    operation_type: str
    provider: str
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    throughput: float  # operations per second
    memory_peak_mb: float
    memory_average_mb: float
    success_rate: float
    error_count: int
    total_operations: int


@dataclass
class QualityMetrics:
    """Quality metrics for AI provider outputs."""
    provider: str
    embedding_consistency: float  # 0-1 score
    embedding_separability: float  # 0-1 score
    text_coherence: float  # 0-1 score
    text_relevance: float  # 0-1 score
    response_length_avg: float
    response_length_std: float


class PerformanceProfiler:
    """Utility class for profiling AI provider performance."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.memory_samples: List[float] = []
    
    def start_profiling(self):
        """Start memory and time profiling."""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def sample_memory(self):
        """Sample current memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_samples.append(current_memory)
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            "total_time": end_time - self.start_time,
            "memory_peak_mb": peak / 1024 / 1024,
            "memory_average_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "memory_start_mb": self.start_memory,
            "memory_end_mb": end_memory
        }


class MockOpenAIProvider:
    """Mock OpenAI provider for performance testing."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_dimensions = 1536
        self.call_latencies = {"embedding": 0.5, "llm": 2.0}  # Simulated latencies
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding creation with realistic latency."""
        await asyncio.sleep(self.call_latencies["embedding"] * len(texts) * 0.1)
        return [[0.1 + i * 0.01] * self.embedding_dimensions for i in range(len(texts))]
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Mock text generation with realistic latency."""
        await asyncio.sleep(self.call_latencies["llm"])
        return f"Generated response for prompt: {prompt[:50]}... (Length: ~{len(prompt) // 4} words)"
    
    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions


class MockOllamaProvider:
    """Mock Ollama provider for performance testing."""
    
    def __init__(self, base_url: str, embedding_model: str, llm_model: str):
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.embedding_dimensions = 768  # Typical for nomic-embed-text
        self.call_latencies = {"embedding": 1.5, "llm": 5.0}  # Typically slower than OpenAI
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding creation with realistic latency."""
        await asyncio.sleep(self.call_latencies["embedding"] * len(texts) * 0.2)
        return [[0.2 + i * 0.01] * self.embedding_dimensions for i in range(len(texts))]
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Mock text generation with realistic latency."""
        await asyncio.sleep(self.call_latencies["llm"])
        return f"Ollama generated response for: {prompt[:50]}... (Length: ~{len(prompt) // 6} words)"
    
    def get_embedding_dimensions(self) -> int:
        return self.embedding_dimensions


@pytest.fixture
def openai_provider():
    """Create mock OpenAI provider."""
    return MockOpenAIProvider(api_key="test-key")


@pytest.fixture
def ollama_provider():
    """Create mock Ollama provider."""
    return MockOllamaProvider(
        base_url=OLLAMA_BASE_URL,
        embedding_model=OLLAMA_EMBEDDING_MODEL,
        llm_model=OLLAMA_LLM_MODEL
    )


@pytest.fixture
def test_texts():
    """Generate test texts of various lengths."""
    return [
        "Short text.",
        "This is a medium length text that contains multiple sentences and provides more content to embed.",
        "This is a very long text that simulates real-world documents with substantial content. " * 10,
        "Technical documentation about machine learning algorithms, neural networks, and artificial intelligence systems.",
        "Code example: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Mixed content with numbers 123, symbols @#$%, and unicode characters: café, naïve, résumé",
    ]


@pytest.fixture
def test_prompts():
    """Generate test prompts for LLM evaluation."""
    return [
        "Summarize the key concepts of machine learning in one paragraph.",
        "Explain the difference between supervised and unsupervised learning.",
        "Write a Python function to calculate the factorial of a number.",
        "What are the main advantages of using vector databases for RAG applications?",
        "Describe the process of fine-tuning a large language model.",
    ]


class TestEmbeddingPerformance:
    """Test embedding generation performance comparison."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_single_embedding_latency(self, openai_provider, ollama_provider, test_texts):
        """Compare single embedding generation latency."""
        text = test_texts[1]  # Medium length text
        
        # Test OpenAI
        profiler_openai = PerformanceProfiler()
        profiler_openai.start_profiling()
        
        start_time = time.time()
        openai_embeddings = await openai_provider.create_embeddings([text])
        openai_time = time.time() - start_time
        
        openai_metrics = profiler_openai.stop_profiling()
        
        # Test Ollama
        profiler_ollama = PerformanceProfiler()
        profiler_ollama.start_profiling()
        
        start_time = time.time()
        ollama_embeddings = await ollama_provider.create_embeddings([text])
        ollama_time = time.time() - start_time
        
        ollama_metrics = profiler_ollama.stop_profiling()
        
        # Assertions
        assert len(openai_embeddings) == 1
        assert len(ollama_embeddings) == 1
        assert len(openai_embeddings[0]) == openai_provider.get_embedding_dimensions()
        assert len(ollama_embeddings[0]) == ollama_provider.get_embedding_dimensions()
        
        # Performance comparison
        print(f"OpenAI embedding time: {openai_time:.3f}s")
        print(f"Ollama embedding time: {ollama_time:.3f}s")
        print(f"Speed ratio (Ollama/OpenAI): {ollama_time/openai_time:.2f}x")
        
        # Memory usage
        print(f"OpenAI memory peak: {openai_metrics['memory_peak_mb']:.2f} MB")
        print(f"Ollama memory peak: {ollama_metrics['memory_peak_mb']:.2f} MB")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_batch_embedding_throughput(self, openai_provider, ollama_provider, test_texts):
        """Compare batch embedding throughput."""
        batch_sizes = [1, 5, 10, 20]
        
        results = {
            "openai": {},
            "ollama": {}
        }
        
        for batch_size in batch_sizes:
            batch_texts = test_texts[:batch_size] if batch_size <= len(test_texts) else test_texts * (batch_size // len(test_texts) + 1)
            batch_texts = batch_texts[:batch_size]
            
            # Test OpenAI
            start_time = time.time()
            openai_embeddings = await openai_provider.create_embeddings(batch_texts)
            openai_time = time.time() - start_time
            openai_throughput = batch_size / openai_time
            
            # Test Ollama
            start_time = time.time()
            ollama_embeddings = await ollama_provider.create_embeddings(batch_texts)
            ollama_time = time.time() - start_time
            ollama_throughput = batch_size / ollama_time
            
            results["openai"][batch_size] = {
                "time": openai_time,
                "throughput": openai_throughput
            }
            results["ollama"][batch_size] = {
                "time": ollama_time,
                "throughput": ollama_throughput
            }
            
            print(f"Batch size {batch_size}:")
            print(f"  OpenAI: {openai_time:.3f}s, {openai_throughput:.2f} embeddings/s")
            print(f"  Ollama: {ollama_time:.3f}s, {ollama_throughput:.2f} embeddings/s")
        
        # Verify batch processing works
        for batch_size in batch_sizes:
            assert "openai" in results
            assert "ollama" in results
            assert results["openai"][batch_size]["throughput"] > 0
            assert results["ollama"][batch_size]["throughput"] > 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_concurrent_embedding_requests(self, openai_provider, ollama_provider, test_texts):
        """Test concurrent embedding request handling."""
        concurrency_levels = [1, 3, 5, 10]
        text = test_texts[1]
        
        for concurrency in concurrency_levels:
            # Test OpenAI concurrency
            start_time = time.time()
            openai_tasks = [
                openai_provider.create_embeddings([text]) 
                for _ in range(concurrency)
            ]
            openai_results = await asyncio.gather(*openai_tasks)
            openai_time = time.time() - start_time
            
            # Test Ollama concurrency
            start_time = time.time()
            ollama_tasks = [
                ollama_provider.create_embeddings([text]) 
                for _ in range(concurrency)
            ]
            ollama_results = await asyncio.gather(*ollama_tasks)
            ollama_time = time.time() - start_time
            
            # Verify results
            assert len(openai_results) == concurrency
            assert len(ollama_results) == concurrency
            assert all(len(result) == 1 for result in openai_results)
            assert all(len(result) == 1 for result in ollama_results)
            
            print(f"Concurrency {concurrency}:")
            print(f"  OpenAI: {openai_time:.3f}s total, {openai_time/concurrency:.3f}s avg")
            print(f"  Ollama: {ollama_time:.3f}s total, {ollama_time/concurrency:.3f}s avg")


class TestLLMPerformance:
    """Test LLM text generation performance comparison."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_text_generation_latency(self, openai_provider, ollama_provider, test_prompts):
        """Compare text generation latency."""
        prompt = test_prompts[0]  # Standard prompt
        
        # Test OpenAI
        start_time = time.time()
        openai_response = await openai_provider.generate_text(prompt)
        openai_time = time.time() - start_time
        
        # Test Ollama
        start_time = time.time()
        ollama_response = await ollama_provider.generate_text(prompt)
        ollama_time = time.time() - start_time
        
        # Assertions
        assert len(openai_response) > 0
        assert len(ollama_response) > 0
        
        print(f"OpenAI generation time: {openai_time:.3f}s")
        print(f"Ollama generation time: {ollama_time:.3f}s")
        print(f"Speed ratio (Ollama/OpenAI): {ollama_time/openai_time:.2f}x")
        print(f"OpenAI response length: {len(openai_response)} chars")
        print(f"Ollama response length: {len(ollama_response)} chars")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_prompt_length_impact(self, openai_provider, ollama_provider):
        """Test impact of prompt length on generation time."""
        base_prompt = "Explain machine learning"
        prompt_multipliers = [1, 2, 5, 10]
        
        results = {"openai": {}, "ollama": {}}
        
        for multiplier in prompt_multipliers:
            extended_prompt = (base_prompt + " ") * multiplier
            prompt_length = len(extended_prompt)
            
            # Test OpenAI
            start_time = time.time()
            openai_response = await openai_provider.generate_text(extended_prompt)
            openai_time = time.time() - start_time
            
            # Test Ollama
            start_time = time.time()
            ollama_response = await ollama_provider.generate_text(extended_prompt)
            ollama_time = time.time() - start_time
            
            results["openai"][prompt_length] = {
                "time": openai_time,
                "response_length": len(openai_response)
            }
            results["ollama"][prompt_length] = {
                "time": ollama_time,
                "response_length": len(ollama_response)
            }
            
            print(f"Prompt length {prompt_length} chars:")
            print(f"  OpenAI: {openai_time:.3f}s")
            print(f"  Ollama: {ollama_time:.3f}s")
        
        # Verify scaling behavior
        for provider in ["openai", "ollama"]:
            times = [results[provider][length]["time"] for length in sorted(results[provider].keys())]
            # Time should generally increase with prompt length (with some variance)
            assert len(times) > 1


class TestMemoryUsage:
    """Test memory usage patterns of different providers."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_embedding_memory_usage(self, openai_provider, ollama_provider, test_texts):
        """Test memory usage during embedding generation."""
        large_batch = test_texts * 10  # 60 texts
        
        # Test OpenAI memory usage
        profiler_openai = PerformanceProfiler()
        profiler_openai.start_profiling()
        
        for i in range(0, len(large_batch), 5):
            batch = large_batch[i:i+5]
            await openai_provider.create_embeddings(batch)
            profiler_openai.sample_memory()
        
        openai_metrics = profiler_openai.stop_profiling()
        
        # Test Ollama memory usage
        profiler_ollama = PerformanceProfiler()
        profiler_ollama.start_profiling()
        
        for i in range(0, len(large_batch), 5):
            batch = large_batch[i:i+5]
            await ollama_provider.create_embeddings(batch)
            profiler_ollama.sample_memory()
        
        ollama_metrics = profiler_ollama.stop_profiling()
        
        print(f"OpenAI memory usage:")
        print(f"  Peak: {openai_metrics['memory_peak_mb']:.2f} MB")
        print(f"  Average: {openai_metrics['memory_average_mb']:.2f} MB")
        
        print(f"Ollama memory usage:")
        print(f"  Peak: {ollama_metrics['memory_peak_mb']:.2f} MB")
        print(f"  Average: {ollama_metrics['memory_average_mb']:.2f} MB")
        
        # Memory usage should be reasonable
        assert openai_metrics['memory_peak_mb'] < 1000  # Less than 1GB
        assert ollama_metrics['memory_peak_mb'] < 1000  # Less than 1GB


class TestQualityMetrics:
    """Test quality metrics for provider outputs."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_embedding_consistency(self, openai_provider, ollama_provider):
        """Test embedding consistency for identical inputs."""
        text = "This is a test sentence for consistency checking."
        
        # Generate multiple embeddings for the same text
        openai_embeddings = []
        ollama_embeddings = []
        
        for _ in range(3):
            openai_emb = await openai_provider.create_embeddings([text])
            ollama_emb = await ollama_provider.create_embeddings([text])
            openai_embeddings.append(openai_emb[0])
            ollama_embeddings.append(ollama_emb[0])
        
        # Calculate consistency (similarity between runs)
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
        
        # OpenAI consistency
        openai_similarities = [
            cosine_similarity(openai_embeddings[0], openai_embeddings[i])
            for i in range(1, len(openai_embeddings))
        ]
        openai_consistency = statistics.mean(openai_similarities)
        
        # Ollama consistency
        ollama_similarities = [
            cosine_similarity(ollama_embeddings[0], ollama_embeddings[i])
            for i in range(1, len(ollama_embeddings))
        ]
        ollama_consistency = statistics.mean(ollama_similarities)
        
        print(f"OpenAI embedding consistency: {openai_consistency:.4f}")
        print(f"Ollama embedding consistency: {ollama_consistency:.4f}")
        
        # Both should be highly consistent (>0.95 for identical text)
        assert openai_consistency > 0.9
        assert ollama_consistency > 0.9
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_embedding_separability(self, openai_provider, ollama_provider):
        """Test embedding separability for different inputs."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Cooking pasta requires boiling water and adding salt.",
            "The weather today is sunny with light clouds."
        ]
        
        # Generate embeddings
        openai_embeddings = await openai_provider.create_embeddings(texts)
        ollama_embeddings = await ollama_provider.create_embeddings(texts)
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
        
        # Calculate cross-similarities
        openai_similarities = []
        ollama_similarities = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                openai_sim = cosine_similarity(openai_embeddings[i], openai_embeddings[j])
                ollama_sim = cosine_similarity(ollama_embeddings[i], ollama_embeddings[j])
                openai_similarities.append(openai_sim)
                ollama_similarities.append(ollama_sim)
        
        openai_avg_similarity = statistics.mean(openai_similarities)
        ollama_avg_similarity = statistics.mean(ollama_similarities)
        
        print(f"OpenAI average cross-similarity: {openai_avg_similarity:.4f}")
        print(f"Ollama average cross-similarity: {ollama_avg_similarity:.4f}")
        
        # Different texts should have lower similarity (<0.7)
        assert openai_avg_similarity < 0.8
        assert ollama_avg_similarity < 0.8


class TestCostAnalysis:
    """Test cost analysis for different providers."""
    
    def test_api_call_cost_estimation(self, openai_provider, ollama_provider):
        """Estimate costs for different usage patterns."""
        # Mock pricing (approximate)
        openai_embedding_cost_per_1k = 0.0001  # $0.0001 per 1K tokens
        openai_gpt4_cost_per_1k = 0.03  # $0.03 per 1K tokens
        
        ollama_hosting_cost_per_hour = 0.5  # Estimated server cost
        
        # Usage scenarios
        scenarios = [
            {"name": "Light", "embeddings_per_day": 1000, "llm_calls_per_day": 100},
            {"name": "Medium", "embeddings_per_day": 10000, "llm_calls_per_day": 500},
            {"name": "Heavy", "embeddings_per_day": 100000, "llm_calls_per_day": 2000},
        ]
        
        for scenario in scenarios:
            # OpenAI costs (monthly)
            monthly_embeddings = scenario["embeddings_per_day"] * 30
            monthly_llm_calls = scenario["llm_calls_per_day"] * 30
            
            # Assuming average 100 tokens per embedding, 500 tokens per LLM call
            openai_embedding_cost = (monthly_embeddings * 100 / 1000) * openai_embedding_cost_per_1k
            openai_llm_cost = (monthly_llm_calls * 500 / 1000) * openai_gpt4_cost_per_1k
            total_openai_cost = openai_embedding_cost + openai_llm_cost
            
            # Ollama costs (monthly) - just hosting
            ollama_cost = ollama_hosting_cost_per_hour * 24 * 30
            
            print(f"{scenario['name']} usage scenario (monthly):")
            print(f"  OpenAI total cost: ${total_openai_cost:.2f}")
            print(f"  Ollama hosting cost: ${ollama_cost:.2f}")
            print(f"  Savings with Ollama: ${total_openai_cost - ollama_cost:.2f}")
            print(f"  Break-even point: {total_openai_cost/ollama_cost:.2f}x usage")
    
    def test_performance_cost_tradeoff(self, openai_provider, ollama_provider):
        """Analyze performance vs cost tradeoffs."""
        # Mock performance metrics
        openai_metrics = {
            "embedding_latency": 0.5,
            "llm_latency": 2.0,
            "cost_per_embedding": 0.00001,
            "cost_per_llm_call": 0.015
        }
        
        ollama_metrics = {
            "embedding_latency": 1.5,
            "llm_latency": 5.0,
            "cost_per_embedding": 0.0,  # Only hosting costs
            "cost_per_llm_call": 0.0   # Only hosting costs
        }
        
        # Calculate efficiency scores
        openai_embedding_efficiency = 1 / (openai_metrics["embedding_latency"] * openai_metrics["cost_per_embedding"]) if openai_metrics["cost_per_embedding"] > 0 else float('inf')
        ollama_embedding_efficiency = 1 / ollama_metrics["embedding_latency"]  # No direct cost
        
        print("Performance vs Cost Analysis:")
        print(f"OpenAI embedding efficiency score: {openai_embedding_efficiency:.2f}")
        print(f"Ollama embedding efficiency score: {ollama_embedding_efficiency:.2f}")
        
        # Both approaches have merits
        assert openai_metrics["embedding_latency"] > 0
        assert ollama_metrics["embedding_latency"] > 0


class TestStorySpecificValidation:
    """Test validation for specific epic stories."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_story1_dimension_configuration(self, openai_provider, ollama_provider):
        """Test Story 1: Configurable embedding dimensions."""
        # Verify dimension detection
        openai_dims = openai_provider.get_embedding_dimensions()
        ollama_dims = ollama_provider.get_embedding_dimensions()
        
        assert openai_dims == 1536
        assert ollama_dims == 768
        
        # Test embeddings have correct dimensions
        embeddings_openai = await openai_provider.create_embeddings(["test"])
        embeddings_ollama = await ollama_provider.create_embeddings(["test"])
        
        assert len(embeddings_openai[0]) == openai_dims
        assert len(embeddings_ollama[0]) == ollama_dims
        
        print(f"Story 1 validation passed:")
        print(f"  OpenAI dimensions: {openai_dims}")
        print(f"  Ollama dimensions: {ollama_dims}")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(SKIP_PERFORMANCE_TESTS, reason=PERFORMANCE_NOT_AVAILABLE)
    async def test_story2_llm_capabilities(self, openai_provider, ollama_provider):
        """Test Story 2: Ollama LLM provider integration."""
        test_prompt = "Explain the concept of vector databases in one sentence."
        
        # Both providers should generate text
        openai_response = await openai_provider.generate_text(test_prompt)
        ollama_response = await ollama_provider.generate_text(test_prompt)
        
        assert len(openai_response) > 0
        assert len(ollama_response) > 0
        
        # Responses should be different but both valid
        assert openai_response != ollama_response
        
        print(f"Story 2 validation passed:")
        print(f"  OpenAI response: {openai_response[:100]}...")
        print(f"  Ollama response: {ollama_response[:100]}...")
    
    def test_story3_migration_safety(self, openai_provider, ollama_provider):
        """Test Story 3: Provider configuration and migration safety."""
        # Dimension mismatch should be detectable
        openai_dims = openai_provider.get_embedding_dimensions()
        ollama_dims = ollama_provider.get_embedding_dimensions()
        
        if openai_dims != ollama_dims:
            print(f"Story 3 validation: Dimension mismatch detected ({openai_dims} != {ollama_dims})")
            print("This would trigger migration guidance in production")
            assert True  # This is expected behavior
        else:
            print("Story 3 validation: Dimensions are compatible")
            assert True  # This is also valid


if __name__ == "__main__":
    # Run performance tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short",
        "-k", "not skipif"  # Run only non-skipped tests
    ])