# AI Provider Testing Guide

This guide provides comprehensive testing procedures for validating AI provider functionality in the MCP Crawl4AI RAG system, covering unit tests, integration tests, and performance validation.

## Testing Overview

The testing framework covers:
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end provider interactions
- **Performance Tests**: Speed and resource usage benchmarks
- **Compatibility Tests**: Cross-provider consistency
- **Deployment Tests**: Docker and orchestration validation

## Quick Testing Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific provider
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider -v
python -m pytest tests/test_ai_providers.py::TestOllamaProvider -v

# Performance comparison
python -m pytest tests/test_performance_comparison.py -v

# Integration testing
python tests/test_runner.py --test-type integration --provider openai
python tests/test_runner.py --test-type integration --provider ollama

# Health check validation
./scripts/health_check.py --provider your_provider --env your_env
```

## Test Environment Setup

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock requests-mock

# Set up test environment variables
export PYTHONPATH="${PYTHONPATH}:."
export PYTEST_TIMEOUT=300

# For OpenAI tests (optional - uses mocks by default)
export OPENAI_API_KEY=your_test_key

# For Ollama tests (requires running Ollama)
export OLLAMA_BASE_URL=http://localhost:11434
./scripts/setup_ollama.py --install --pull-models
```

### Test Configuration Files

Create test-specific environment files:

```bash
# tests/.env.test.openai
AI_PROVIDER=openai
OPENAI_API_KEY=sk-test-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/tmp/test_vector_db.sqlite

# tests/.env.test.ollama
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:1b
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/tmp/test_vector_db.sqlite

# tests/.env.test.hybrid
AI_PROVIDER=mixed
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=ollama
OPENAI_API_KEY=sk-test-key
OLLAMA_BASE_URL=http://localhost:11434
VECTOR_DB_PROVIDER=sqlite
SQLITE_DB_PATH=/tmp/test_vector_db.sqlite
```

## Unit Tests

### AI Provider Tests

**Test OpenAI Provider:**
```bash
# Basic functionality
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider::test_create_embedding -v

# Configuration validation
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider::test_configuration_validation -v

# Error handling
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider::test_error_handling -v

# Batch processing  
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider::test_batch_embedding_creation -v
```

**Test Ollama Provider:**
```bash
# Basic functionality (requires Ollama running)
python -m pytest tests/test_ai_providers.py::TestOllamaProvider::test_create_embedding -v

# Model availability
python -m pytest tests/test_ai_providers.py::TestOllamaProvider::test_model_availability -v

# Connection handling
python -m pytest tests/test_ai_providers.py::TestOllamaProvider::test_connection_handling -v
```

**Test Provider Factory:**
```bash
# Provider selection
python -m pytest tests/test_ai_providers.py::TestProviderFactory::test_provider_selection -v

# Dynamic switching
python -m pytest tests/test_ai_providers.py::TestProviderFactory::test_dynamic_provider_switching -v
```

### Utils Refactored Tests

```bash
# Test backward compatibility
python -m pytest tests/test_utils_refactored.py::TestBackwardCompatibility -v

# Test provider abstraction
python -m pytest tests/test_utils_refactored.py::TestProviderAbstraction -v

# Test error handling
python -m pytest tests/test_utils_refactored.py::TestErrorHandling -v

# Test performance optimizations
python -m pytest tests/test_utils_refactored.py::TestPerformanceOptimizations -v
```

### Database Provider Tests

```bash
# Test all database providers
python -m pytest tests/test_database_providers.py -v

# Test specific provider
python -m pytest tests/test_database_providers.py::TestSupabaseProvider -v
python -m pytest tests/test_database_providers.py::TestSQLiteProvider -v
```

## Integration Tests

### End-to-End Testing

**Test Complete Workflows:**
```bash
# Full crawl and RAG workflow
python -m pytest tests/test_comprehensive_integration.py::test_full_crawl_and_rag_workflow -v

# Multi-provider consistency
python -m pytest tests/test_comprehensive_integration.py::test_provider_consistency -v

# RAG strategy combinations
python -m pytest tests/test_comprehensive_integration.py::test_rag_strategy_combinations -v
```

**Test MCP Server Integration:**
```bash
# MCP tool functionality
python -m pytest tests/test_epic_stories.py::test_crawl_and_query_workflow -v

# Provider switching during runtime
python -m pytest tests/test_epic_stories.py::test_provider_switching -v
```

### Using Test Runner

The test runner provides automated testing across different configurations:

```bash
# Test OpenAI configuration
python tests/test_runner.py --test-type integration --provider openai --env test

# Test Ollama configuration (requires Ollama)
python tests/test_runner.py --test-type integration --provider ollama --env test

# Test hybrid configuration
python tests/test_runner.py --test-type integration --provider mixed --env test

# Test specific scenarios
python tests/test_runner.py --test-type performance --provider openai
python tests/test_runner.py --test-type migration --provider ollama
```

## Performance Tests

### Benchmark Testing

```bash
# Run performance comparison
python -m pytest tests/test_performance_comparison.py::test_embedding_performance_comparison -v

# Test with different batch sizes
python -m pytest tests/test_performance_comparison.py::test_batch_size_optimization -v

# Memory usage testing
python -m pytest tests/test_performance_comparison.py::test_memory_usage -v

# Concurrent request testing
python -m pytest tests/test_performance_comparison.py::test_concurrent_requests -v
```

### Custom Performance Testing

```python
# Custom performance test example
import time
from utils_refactored import create_embeddings_batch

def test_performance_custom():
    texts = ["test text"] * 100
    
    # Warm up
    create_embeddings_batch(texts[:10])
    
    # Measure performance
    start_time = time.time()
    embeddings = create_embeddings_batch(texts)
    end_time = time.time()
    
    duration = end_time - start_time
    rate = len(texts) / duration
    
    print(f"Processed {len(texts)} texts in {duration:.2f}s")
    print(f"Rate: {rate:.1f} texts/second")
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) > 0 for emb in embeddings)
    assert rate > 10  # Minimum acceptable rate

if __name__ == "__main__":
    test_performance_custom()
```

### Load Testing

```bash
# Stress test with high volume
python -c "
import asyncio
from utils_refactored import create_embeddings_batch

async def load_test():
    texts = ['test'] * 1000
    batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
    
    tasks = []
    for batch in batches:
        task = asyncio.create_task(
            asyncio.to_thread(create_embeddings_batch, batch)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    total_embeddings = sum(len(result) for result in results)
    print(f'Processed {total_embeddings} embeddings successfully')

asyncio.run(load_test())
"
```

## Provider Compatibility Tests

### Cross-Provider Consistency

Test that different providers produce consistent results:

```bash
# Run compatibility tests
python -m pytest tests/test_migration_integrity.py::test_provider_consistency -v

# Test dimension compatibility
python -m pytest tests/test_migration_integrity.py::test_embedding_dimensions -v

# Test quality consistency
python -m pytest tests/test_migration_integrity.py::test_semantic_similarity_consistency -v
```

### Migration Testing

```bash
# Test migration between providers
python -m pytest tests/test_migration_integrity.py::test_openai_to_ollama_migration -v

# Test hybrid configuration migration
python -m pytest tests/test_migration_integrity.py::test_hybrid_migration -v

# Test rollback procedures
python -m pytest tests/test_migration_integrity.py::test_migration_rollback -v
```

## Docker and Deployment Tests

### Container Testing

```bash
# Test Docker builds
python -m pytest tests/test_docker_integration.py::test_docker_build -v

# Test multi-provider containers
python -m pytest tests/test_docker_integration.py::test_multi_provider_containers -v

# Test health checks
python -m pytest tests/test_docker_integration.py::test_container_health_checks -v
```

### Orchestration Testing

```bash
# Test Docker Compose profiles
python -m pytest tests/test_docker_orchestration.py::test_compose_profiles -v

# Test service dependencies
python -m pytest tests/test_docker_orchestration.py::test_service_dependencies -v

# Test networking
python -m pytest tests/test_docker_orchestration.py::test_container_networking -v
```

### Using Test Containers

The project includes testcontainers integration for isolated testing:

```bash
# Test with isolated containers
python -m pytest tests/test_testcontainers.py -v

# Test Ollama container setup
python -m pytest tests/test_testcontainers.py::test_ollama_container -v

# Test Neo4j container setup
python -m pytest tests/test_testcontainers.py::test_neo4j_container -v
```

## Manual Testing Procedures

### Provider Validation

**1. OpenAI Provider Validation:**
```bash
# Test configuration
export AI_PROVIDER=openai
export OPENAI_API_KEY=your_key

# Test basic functionality
python -c "
from utils_refactored import create_embedding, validate_ai_provider_config
print('Config valid:', validate_ai_provider_config()['valid'])
embedding = create_embedding('Hello world')
print(f'Embedding created: {len(embedding)} dimensions')
"

# Test batch processing
python -c "
from utils_refactored import create_embeddings_batch
texts = ['hello', 'world', 'test', 'batch']
embeddings = create_embeddings_batch(texts)
print(f'Batch processed: {len(embeddings)} embeddings')
"
```

**2. Ollama Provider Validation:**
```bash
# Ensure Ollama is running
./scripts/setup_ollama.py --status

# Test configuration
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434

# Test model availability
./scripts/setup_ollama.py --validate

# Test functionality
python -c "
from utils_refactored import create_embedding
embedding = create_embedding('Hello world')
print(f'Ollama embedding: {len(embedding)} dimensions')
"
```

**3. Hybrid Configuration Validation:**
```bash
# Set hybrid configuration
export AI_PROVIDER=mixed
export EMBEDDING_PROVIDER=openai
export LLM_PROVIDER=ollama

# Test both components
python -c "
from utils_refactored import create_embedding, generate_contextual_embedding
embedding = create_embedding('Test embedding')
print(f'Embedding (OpenAI): {len(embedding)} dimensions')

context, success = generate_contextual_embedding('Full document', 'Chunk text')
print(f'Contextual generation (Ollama): {success}')
"
```

### MCP Server Testing

**1. Start MCP Server:**
```bash
# With specific provider
export AI_PROVIDER=your_provider
uv run src/crawl4ai_mcp.py
```

**2. Test MCP Tools:**
```bash
# Using curl (if SSE mode)
curl -X POST http://localhost:8051/mcp/tools/crawl_single_page \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Using MCP client (recommended)
# Connect your MCP client to test tools:
# - crawl_single_page
# - perform_rag_query
# - get_available_sources
```

## Automated Testing Workflows

### GitHub Actions Integration

Example workflow for continuous testing:

```yaml
# .github/workflows/test-ai-providers.yml
name: AI Provider Tests

on: [push, pull_request]

jobs:
  test-openai:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio
      - name: Test OpenAI Provider (mocked)
        run: |
          pytest tests/test_ai_providers.py::TestOpenAIProvider -v
        env:
          AI_PROVIDER: openai
          OPENAI_API_KEY: sk-test-key

  test-ollama:
    runs-on: ubuntu-latest
    services:
      ollama:
        image: ollama/ollama:latest
        ports:
          - 11434:11434
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.11'
      - name: Pull Ollama models
        run: |
          docker exec ollama ollama pull nomic-embed-text
          docker exec ollama ollama pull llama3.2:1b
      - name: Test Ollama Provider
        run: |
          pytest tests/test_ai_providers.py::TestOllamaProvider -v
        env:
          AI_PROVIDER: ollama
          OLLAMA_BASE_URL: http://localhost:11434
```

### Local Testing Scripts

**Create comprehensive test script:**
```bash
#!/bin/bash
# test_all_providers.sh

set -e

echo "=== Testing AI Providers ==="

# Test OpenAI (mocked)
echo "Testing OpenAI provider..."
export AI_PROVIDER=openai
export OPENAI_API_KEY=sk-test-key
python -m pytest tests/test_ai_providers.py::TestOpenAIProvider -v

# Test Ollama (if available)
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Testing Ollama provider..."
    export AI_PROVIDER=ollama
    export OLLAMA_BASE_URL=http://localhost:11434
    python -m pytest tests/test_ai_providers.py::TestOllamaProvider -v
else
    echo "Skipping Ollama tests (service not available)"
fi

# Test integration
echo "Testing integration..."
python -m pytest tests/test_comprehensive_integration.py -v

# Test performance
echo "Testing performance..."
python -m pytest tests/test_performance_comparison.py -v

echo "=== All tests completed ==="
```

## Test Reporting and Analysis

### Generate Test Reports

```bash
# Generate detailed test report
python -m pytest tests/ -v --tb=short --html=reports/test_report.html

# Generate coverage report
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html:reports/coverage

# Generate performance report
python tests/test_runner.py --test-type performance --provider all --output reports/performance.json
```

### Analyze Test Results

```python
# analyze_test_results.py
import json
import matplotlib.pyplot as plt

def analyze_performance_results(report_file):
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    providers = []
    embedding_times = []
    
    for provider, metrics in data.items():
        providers.append(provider)
        embedding_times.append(metrics['embedding_time'])
    
    plt.figure(figsize=(10, 6))
    plt.bar(providers, embedding_times)
    plt.title('Embedding Performance by Provider')
    plt.ylabel('Time (seconds)')
    plt.savefig('reports/performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    analyze_performance_results('reports/performance.json')
```

## Troubleshooting Test Issues

### Common Test Failures

**1. Ollama Connection Issues:**
```bash
# Check if Ollama is running
./scripts/setup_ollama.py --status

# Start Ollama if needed
ollama serve &
sleep 10  # Wait for startup

# Pull required models
./scripts/setup_ollama.py --pull-models
```

**2. API Key Issues:**
```bash
# Use mock for testing (default)
unset OPENAI_API_KEY

# Or use real API key for integration tests
export OPENAI_API_KEY=your_real_key
```

**3. Database Issues:**
```bash
# Use temporary database for tests
export SQLITE_DB_PATH=/tmp/test_$(date +%s).sqlite

# Clean up test databases
rm -f /tmp/test_*.sqlite
```

### Test Environment Isolation

```bash
# Use virtual environment for testing
python -m venv test_env
source test_env/bin/activate
pip install -e .
pip install pytest pytest-asyncio

# Run tests with clean environment
./test_all_providers.sh

# Deactivate when done
deactivate
```

## Quality Assurance Checklist

Before releasing or deploying:

- [ ] All unit tests pass for each provider
- [ ] Integration tests pass for common workflows
- [ ] Performance benchmarks meet acceptable thresholds
- [ ] Docker builds complete successfully
- [ ] Health checks validate correctly
- [ ] Migration tests pass
- [ ] Documentation is updated with test results
- [ ] Error handling works correctly
- [ ] Resource usage is within acceptable limits
- [ ] Cross-provider consistency is maintained

## Custom Test Development

When adding new features or providers:

1. **Write unit tests first** (TDD approach)
2. **Add integration tests** for new workflows
3. **Include performance benchmarks** for new providers
4. **Update test documentation** with new procedures
5. **Add to CI/CD pipeline** for automated testing

This testing guide ensures comprehensive validation of all AI provider functionality and helps maintain high quality across different deployment scenarios.