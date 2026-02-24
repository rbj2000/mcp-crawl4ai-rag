"""
Pytest configuration and shared fixtures for the Ollama AI Provider testing framework.

This module provides:
1. Common test configuration and fixtures
2. Test environment setup and teardown
3. Mock providers and test data generators
4. Test utilities and helper functions
5. Parametrized test configurations
"""

import pytest
import asyncio
import os
import tempfile
import json
from typing import List, Dict, Any, Optional
import logging

# Configure logging for tests  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration from environment
TEST_CONFIG = {
    "enable_unit_tests": os.getenv("ENABLE_UNIT_TESTS", "true").lower() == "true",
    "enable_integration_tests": os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true", 
    "enable_performance_tests": os.getenv("ENABLE_PERFORMANCE_TESTS", "false").lower() == "true",
    "enable_migration_tests": os.getenv("ENABLE_MIGRATION_TESTS", "false").lower() == "true",
    "enable_docker_tests": os.getenv("ENABLE_DOCKER_TESTS", "false").lower() == "true",
    "enable_story_validation": os.getenv("ENABLE_STORY_VALIDATION", "true").lower() == "true",
    
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "ollama_embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    "ollama_llm_model": os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
    "openai_api_key": os.getenv("OPENAI_API_KEY", "test-key"),
    
    "test_timeout": int(os.getenv("TEST_TIMEOUT", "300")),  # 5 minutes default
    "parallel_tests": int(os.getenv("PYTEST_WORKERS", "4")),
}


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "migration: Migration tests")
    config.addinivalue_line("markers", "docker: Docker tests")
    config.addinivalue_line("markers", "story: Story validation tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_ollama: Tests requiring Ollama server")
    config.addinivalue_line("markers", "requires_docker: Tests requiring Docker")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    for item in items:
        # Add markers based on test file names
        if "test_ai_providers" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_ollama_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.requires_ollama)
        elif "test_performance_comparison" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "test_migration_integrity" in item.nodeid:
            item.add_marker(pytest.mark.migration)
        elif "test_docker_orchestration" in item.nodeid:
            item.add_marker(pytest.mark.docker)
            item.add_marker(pytest.mark.requires_docker)
        elif "test_epic_stories" in item.nodeid:
            item.add_marker(pytest.mark.story)
        elif "test_reranking_providers" in item.nodeid:
            item.add_marker(pytest.mark.reranking)
            item.add_marker(pytest.mark.reranking_unit)
            item.add_marker(pytest.mark.unit)
        elif "test_reranking_integration" in item.nodeid:
            item.add_marker(pytest.mark.reranking)
            item.add_marker(pytest.mark.reranking_integration)
            item.add_marker(pytest.mark.integration)
        
        # Add skip conditions based on configuration
        if item.get_closest_marker("integration") and not TEST_CONFIG["enable_integration_tests"]:
            item.add_marker(pytest.mark.skip(reason="Integration tests disabled"))
        elif item.get_closest_marker("performance") and not TEST_CONFIG["enable_performance_tests"]:
            item.add_marker(pytest.mark.skip(reason="Performance tests disabled"))
        elif item.get_closest_marker("migration") and not TEST_CONFIG["enable_migration_tests"]:
            item.add_marker(pytest.mark.skip(reason="Migration tests disabled"))
        elif item.get_closest_marker("docker") and not TEST_CONFIG["enable_docker_tests"]:
            item.add_marker(pytest.mark.skip(reason="Docker tests disabled"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture."""
    return TEST_CONFIG


@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="ollama_test_")
    yield temp_dir
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def openai_provider_config():
    """OpenAI provider configuration fixture."""
    return {
        "provider": "openai",
        "api_key": TEST_CONFIG["openai_api_key"],
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": 1536,
        "llm_model": "gpt-4",
        "max_retries": 3,
        "timeout": 30
    }


@pytest.fixture
def ollama_provider_config():
    """Ollama provider configuration fixture."""
    return {
        "provider": "ollama", 
        "base_url": TEST_CONFIG["ollama_base_url"],
        "embedding_model": TEST_CONFIG["ollama_embedding_model"],
        "embedding_dimensions": 768,
        "llm_model": TEST_CONFIG["ollama_llm_model"],
        "timeout": 60,
        "verify_ssl": False
    }


@pytest.fixture(params=[384, 768, 1024, 1536])
def embedding_dimensions(request):
    """Parametrized fixture for different embedding dimensions."""
    return request.param


@pytest.fixture(params=["sqlite", "memory"])
def database_provider_type(request):
    """Parametrized fixture for database provider types."""
    return request.param


@pytest.fixture
def database_configs():
    """Database configuration fixtures."""
    return {
        "sqlite_1536": {
            "provider": "sqlite",
            "db_path": ":memory:",
            "embedding_dimension": 1536
        },
        "sqlite_768": {
            "provider": "sqlite",
            "db_path": ":memory:",
            "embedding_dimension": 768
        },
        "sqlite_384": {
            "provider": "sqlite",
            "db_path": ":memory:",
            "embedding_dimension": 384
        }
    }


@pytest.fixture
def test_documents():
    """Test documents fixture for validation."""
    return [
        {
            "url": "https://docs.python.org/3/tutorial/",
            "content": "Python is an easy to learn, powerful programming language with efficient high-level data structures and a simple but effective approach to object-oriented programming.",
            "metadata": {"category": "documentation", "language": "python", "difficulty": "beginner"},
            "expected_embedding_quality": 0.8,
            "source_id": "python.org"
        },
        {
            "url": "https://www.tensorflow.org/guide/keras",
            "content": "Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.",
            "metadata": {"category": "documentation", "language": "python", "topic": "ml", "framework": "keras"},
            "expected_embedding_quality": 0.8,
            "source_id": "tensorflow.org"
        },
        {
            "url": "https://github.com/example/algorithms",
            "content": "def fibonacci(n):\n    \"\"\"Calculate nth Fibonacci number using recursion.\"\"\"\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "metadata": {"category": "code", "language": "python", "algorithm": "fibonacci", "complexity": "O(2^n)"},
            "expected_embedding_quality": 0.7,
            "source_id": "github.com"
        },
        {
            "url": "https://numpy.org/doc/stable/user/quickstart.html",
            "content": "NumPy is the fundamental package for scientific computing with Python. It contains among other things: a powerful N-dimensional array object, sophisticated broadcasting functions, tools for integrating C/C++ and Fortran code.",
            "metadata": {"category": "documentation", "language": "python", "topic": "scientific", "library": "numpy"},
            "expected_embedding_quality": 0.8,
            "source_id": "numpy.org"
        },
        {
            "url": "https://fastapi.tiangolo.com/tutorial/",
            "content": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. The key features are: Fast, Easy, Robust, Standards-based.",
            "metadata": {"category": "documentation", "language": "python", "topic": "web", "framework": "fastapi"},
            "expected_embedding_quality": 0.8,
            "source_id": "fastapi.tiangolo.com"
        }
    ]


@pytest.fixture
def test_code_examples():
    """Test code examples fixture."""
    return [
        {
            "url": "https://github.com/example/ml-algorithms",
            "content": "class LinearRegression:\n    def __init__(self):\n        self.weights = None\n        self.bias = None\n    \n    def fit(self, X, y):\n        # Implementation of linear regression\n        pass",
            "summary": "Linear regression class implementation",
            "metadata": {"language": "python", "topic": "ml", "algorithm": "linear_regression"},
            "source_id": "github.com"
        },
        {
            "url": "https://github.com/example/data-structures",
            "content": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "summary": "Binary search algorithm implementation",
            "metadata": {"language": "python", "topic": "algorithms", "complexity": "O(log n)"},
            "source_id": "github.com"
        },
        {
            "url": "https://github.com/example/async-python",
            "content": "import asyncio\n\nasync def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.text()\n\nasync def main():\n    urls = ['http://example1.com', 'http://example2.com']\n    tasks = [fetch_data(url) for url in urls]\n    results = await asyncio.gather(*tasks)\n    return results",
            "summary": "Async HTTP requests with aiohttp",
            "metadata": {"language": "python", "topic": "async", "library": "aiohttp"},
            "source_id": "github.com"
        }
    ]


@pytest.fixture
def test_prompts():
    """Test prompts fixture for LLM validation."""
    return [
        {
            "prompt": "Explain the concept of vector embeddings in machine learning in one paragraph.",
            "expected_length_min": 100,
            "expected_length_max": 500,
            "required_keywords": ["vector", "embedding", "similarity", "machine learning"],
            "category": "technical_explanation"
        },
        {
            "prompt": "Write a Python function to calculate cosine similarity between two vectors.",
            "expected_length_min": 50,
            "expected_length_max": 300,
            "required_keywords": ["def", "cosine", "similarity", "vector", "dot product"],
            "category": "code_generation"
        },
        {
            "prompt": "Summarize the advantages and disadvantages of using local AI models versus cloud APIs.",
            "expected_length_min": 100,
            "expected_length_max": 400,
            "required_keywords": ["local", "cloud", "advantage", "disadvantage"],
            "category": "comparison_analysis"
        },
        {
            "prompt": "Given the following context about neural networks, explain the backpropagation algorithm: Neural networks learn by adjusting weights through gradient descent.",
            "expected_length_min": 80,
            "expected_length_max": 350,
            "required_keywords": ["backpropagation", "gradient", "weights", "neural"],
            "category": "contextual_explanation"
        },
        {
            "prompt": "Create a short story about an AI that learns to understand human emotions.",
            "expected_length_min": 150,
            "expected_length_max": 600,
            "required_keywords": ["AI", "emotions", "learn", "human"],
            "category": "creative_writing"
        }
    ]


@pytest.fixture
def performance_test_data():
    """Performance test data fixture."""
    return {
        "small_batch": ["Short text"] * 5,
        "medium_batch": ["This is a medium length text that contains multiple sentences and provides more content to process for embedding generation."] * 20,
        "large_batch": ["This is a very long text that simulates real-world documents with substantial content. " * 10] * 100,
        "mixed_batch": [
            "Short",
            "Medium length text with several words",
            "Very long text that goes on and on with lots of content " * 5,
            "Code: def example(): return True",
            "Technical documentation about machine learning algorithms and neural networks with detailed explanations."
        ] * 10
    }


@pytest.fixture
def docker_compose_config():
    """Docker Compose configuration for testing."""
    return {
        "version": "3.8",
        "services": {
            "mcp-server": {
                "image": "mcp/crawl4ai-rag:test",
                "ports": ["8051:8051"],
                "environment": {
                    "AI_PROVIDER": "ollama",
                    "OLLAMA_BASE_URL": "http://ollama:11434",
                    "DATABASE_PROVIDER": "sqlite",
                    "OPENAI_API_KEY": "fallback-key"
                },
                "depends_on": ["ollama"],
                "networks": ["mcp-network"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8051/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            },
            "ollama": {
                "image": "ollama/ollama:latest",
                "ports": ["11434:11434"],
                "environment": {"OLLAMA_HOST": "0.0.0.0"},
                "volumes": ["ollama-data:/root/.ollama"],
                "networks": ["mcp-network"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:11434/api/tags"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
        },
        "volumes": {"ollama-data": {}},
        "networks": {"mcp-network": {"driver": "bridge"}}
    }


@pytest.fixture
def migration_test_scenarios():
    """Migration test scenarios fixture."""
    return [
        {
            "name": "openai_to_ollama",
            "source": {"provider": "openai", "dimensions": 1536},
            "target": {"provider": "ollama", "dimensions": 768},
            "strategy": "full_recompute",
            "expected_issues": ["dimension_mismatch", "provider_change"]
        },
        {
            "name": "ollama_to_openai", 
            "source": {"provider": "ollama", "dimensions": 768},
            "target": {"provider": "openai", "dimensions": 1536},
            "strategy": "full_recompute",
            "expected_issues": ["dimension_mismatch", "provider_change"]
        },
        {
            "name": "dimension_upgrade",
            "source": {"provider": "ollama", "dimensions": 384},
            "target": {"provider": "ollama", "dimensions": 768},
            "strategy": "full_recompute",
            "expected_issues": ["dimension_mismatch"]
        },
        {
            "name": "compatible_switch",
            "source": {"provider": "openai", "dimensions": 1536},
            "target": {"provider": "openai", "dimensions": 1536},
            "strategy": "no_migration",
            "expected_issues": []
        }
    ]


# Utility functions for tests

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def generate_mock_embedding(dimensions: int, seed: Optional[int] = None) -> List[float]:
    """Generate a mock embedding with specified dimensions."""
    import random
    if seed is not None:
        random.seed(seed)
    
    # Generate normalized random vector
    embedding = [random.gauss(0, 1) for _ in range(dimensions)]
    norm = sum(x * x for x in embedding) ** 0.5
    
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding


def validate_embedding_quality(embedding: List[float], expected_quality: float = 0.7) -> bool:
    """Validate embedding quality based on basic metrics."""
    if not embedding:
        return False
    
    # Check for non-zero values
    non_zero_ratio = sum(1 for x in embedding if abs(x) > 1e-10) / len(embedding)
    
    # Check for reasonable value distribution
    import statistics
    if len(embedding) > 1:
        std_dev = statistics.stdev(embedding)
        mean_abs = statistics.mean(abs(x) for x in embedding)
        
        # Basic quality checks
        quality_score = min(non_zero_ratio, std_dev, mean_abs) 
        return quality_score >= expected_quality
    
    return non_zero_ratio > 0.5


def create_test_database_config(provider_type: str, dimensions: int, temp_dir: str) -> Dict[str, Any]:
    """Create test database configuration."""
    if provider_type == "sqlite":
        return {
            "provider": "sqlite",
            "db_path": os.path.join(temp_dir, f"test_{dimensions}d.sqlite"),
            "embedding_dimension": dimensions
        }
    elif provider_type == "memory":
        return {
            "provider": "sqlite",
            "db_path": ":memory:",
            "embedding_dimension": dimensions
        }
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")


# Test markers and fixtures for story validation

@pytest.fixture
def story1_acceptance_criteria():
    """Story 1 acceptance criteria fixture."""
    return [
        {"id": "S1-AC1", "description": "Configurable embedding dimensions", "priority": "must"},
        {"id": "S1-AC2", "description": "Ollama embedding generation", "priority": "must"},
        {"id": "S1-AC3", "description": "Dimension validation", "priority": "must"},
        {"id": "S1-AC4", "description": "Variable dimension storage", "priority": "must"},
        {"id": "S1-AC5", "description": "Acceptable performance", "priority": "should"}
    ]


@pytest.fixture
def story2_acceptance_criteria():
    """Story 2 acceptance criteria fixture."""
    return [
        {"id": "S2-AC1", "description": "Coherent text generation", "priority": "must"},
        {"id": "S2-AC2", "description": "RAG capability parity", "priority": "must"},
        {"id": "S2-AC3", "description": "Error handling", "priority": "must"},
        {"id": "S2-AC4", "description": "Contextual responses", "priority": "should"}
    ]


@pytest.fixture
def story3_acceptance_criteria():
    """Story 3 acceptance criteria fixture."""
    return [
        {"id": "S3-AC1", "description": "Seamless provider switching", "priority": "must"},
        {"id": "S3-AC2", "description": "Dimension validation during switch", "priority": "must"},
        {"id": "S3-AC3", "description": "Regression test coverage", "priority": "must"},
        {"id": "S3-AC4", "description": "Docker Ollama integration", "priority": "should"},
        {"id": "S3-AC5", "description": "Migration guidance", "priority": "should"}
    ]


# Reranking-specific fixtures

@pytest.fixture
def reranking_provider_configs():
    """Reranking provider configurations fixture."""
    return {
        "openai": {
            "provider": "openai",
            "api_key": "sk-test-key-reranking",
            "embedding_model": "text-embedding-3-small",
            "reranking_method": "similarity_based"
        },
        "ollama": {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "reranking_model": "bge-reranker-base",
            "timeout": 120
        },
        "huggingface": {
            "provider": "huggingface",
            "reranking_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cpu",
            "batch_size": 32
        }
    }


@pytest.fixture
def reranking_test_queries():
    """Test queries for reranking evaluation."""
    return [
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision techniques",
        "artificial intelligence systems",
        "data science methods",
        "software engineering practices",
        "database optimization",
        "web development frameworks",
        "cloud computing platforms"
    ]


@pytest.fixture
def reranking_test_documents():
    """Test documents for reranking evaluation."""
    return [
        {
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "url": "https://example.com/ml-intro",
            "metadata": {"topic": "machine_learning", "difficulty": "beginner", "type": "introduction"}
        },
        {
            "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "url": "https://example.com/deep-learning",
            "metadata": {"topic": "deep_learning", "difficulty": "intermediate", "type": "technical"}
        },
        {
            "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "url": "https://example.com/nlp",
            "metadata": {"topic": "nlp", "difficulty": "intermediate", "type": "definition"}
        },
        {
            "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
            "url": "https://example.com/computer-vision",
            "metadata": {"topic": "computer_vision", "difficulty": "advanced", "type": "overview"}
        },
        {
            "content": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of intelligent agents.",
            "url": "https://example.com/ai-overview",
            "metadata": {"topic": "artificial_intelligence", "difficulty": "beginner", "type": "overview"}
        }
    ]


@pytest.fixture
def reranking_environment_configs():
    """Environment configurations for reranking tests."""
    return {
        "reranking_enabled": {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-key"
        },
        "reranking_disabled": {
            "USE_RERANKING": "false"
        },
        "ollama_reranking": {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "ollama",
            "OLLAMA_BASE_URL": "http://localhost:11434",
            "OLLAMA_RERANKING_MODEL": "bge-reranker-base"
        },
        "huggingface_reranking": {
            "USE_RERANKING": "true",
            "RERANKING_PROVIDER": "huggingface",
            "HF_RERANKING_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "HF_DEVICE": "cpu"
        },
        "mixed_strategies": {
            "USE_RERANKING": "true",
            "USE_CONTEXTUAL_EMBEDDINGS": "true",
            "USE_HYBRID_SEARCH": "true",
            "RERANKING_PROVIDER": "openai"
        }
    }