# Reranking Testing Guide

This guide covers the comprehensive test suite for reranking functionality in the MCP Crawl4AI RAG system.

## Overview

The reranking test suite ensures that:
1. Individual reranking providers work correctly (OpenAI, Ollama, HuggingFace)
2. RAG pipeline integration with reranking is robust
3. Configuration and environment handling is reliable
4. Error scenarios are handled gracefully
5. Existing functionality remains unaffected (regression testing)

## Test Structure

### Test Files

- **`test_reranking_providers.py`** - Unit tests for individual reranking providers
- **`test_reranking_integration.py`** - Integration tests with RAG pipeline
- **`conftest.py`** - Shared fixtures and configuration
- **`run_reranking_tests.py`** - Comprehensive test runner

### Test Categories

1. **Unit Tests** (`@pytest.mark.reranking_unit`)
   - Individual provider functionality
   - Input validation
   - Error handling
   - Performance characteristics

2. **Integration Tests** (`@pytest.mark.reranking_integration`)
   - RAG pipeline with reranking enabled/disabled
   - Provider switching
   - Environment configuration
   - Compatibility with existing RAG strategies

3. **Configuration Tests** (`@pytest.mark.reranking_config`)
   - Environment variable handling
   - Provider configuration validation
   - Fallback mechanisms

4. **Error Handling Tests** (`@pytest.mark.reranking_errors`)
   - Network failures
   - API rate limits
   - Malformed responses
   - Graceful degradation

5. **Regression Tests** (`@pytest.mark.reranking_regression`)
   - Backward compatibility
   - Existing functionality preservation
   - Performance impact analysis

## Running Tests

### Quick Start

```bash
# Run all reranking tests
python run_reranking_tests.py --mode full

# Run only unit tests (fastest)
python run_reranking_tests.py --mode unit

# Run integration tests
python run_reranking_tests.py --mode integration
```

### Specific Test Categories

```bash
# Configuration tests
python run_reranking_tests.py --mode config

# Error handling tests  
python run_reranking_tests.py --mode errors

# Regression tests
python run_reranking_tests.py --mode regression

# Performance benchmarks
python run_reranking_tests.py --mode performance

# Coverage analysis
python run_reranking_tests.py --mode coverage
```

### Using pytest directly

```bash
# Run all reranking tests
pytest tests/test_reranking_*.py -m reranking -v

# Run only unit tests
pytest tests/test_reranking_providers.py -m reranking_unit -v

# Run only integration tests
pytest tests/test_reranking_integration.py -m reranking_integration -v

# Run specific test class
pytest tests/test_reranking_providers.py::TestOpenAIRerankingProvider -v

# Run with coverage
pytest tests/test_reranking_*.py --cov=src/ai_providers --cov-report=html
```

## Test Environment Setup

### Environment Variables

The tests use environment variables for configuration:

```bash
# Enable/disable reranking
export USE_RERANKING=true

# Provider selection
export RERANKING_PROVIDER=openai|ollama|huggingface

# Provider-specific configuration
export OPENAI_API_KEY=sk-your-key
export OLLAMA_BASE_URL=http://localhost:11434
export HF_DEVICE=cpu

# Test-specific settings
export USE_MOCK_PROVIDERS=true
export RERANKING_TIMEOUT=30
```

### Dependencies

Required packages for testing:
- `pytest>=7.0.0`
- `pytest-asyncio>=0.21.0`
- `pytest-cov>=4.0.0`
- `asyncio` (built-in)

Optional for real provider testing:
- `openai>=1.0.0`
- `aiohttp>=3.8.0`  
- `sentence-transformers>=2.2.0`

## Test Fixtures

### Shared Fixtures (conftest.py)

- **`reranking_provider_configs`** - Provider configuration templates
- **`reranking_test_queries`** - Standard test queries
- **`reranking_test_documents`** - Sample documents for testing
- **`reranking_environment_configs`** - Environment variable sets

### Example Usage

```python
def test_provider_configuration(reranking_provider_configs):
    """Test using shared provider configurations."""
    openai_config = reranking_provider_configs["openai"]
    assert openai_config["provider"] == "openai"
    assert "api_key" in openai_config

@pytest.mark.asyncio
async def test_document_reranking(reranking_test_queries, reranking_test_documents):
    """Test using shared test data."""
    query = reranking_test_queries[0]
    documents = reranking_test_documents
    # Test reranking logic...
```

## Provider-Specific Testing

### OpenAI Provider Tests

Tests similarity-based reranking using embeddings:
- Cosine similarity calculation
- Embedding batch processing
- API rate limiting
- Error handling and retries

```python
class TestOpenAIRerankingProvider:
    async def test_similarity_reranking(self):
        # Test OpenAI similarity-based reranking
        pass
```

### Ollama Provider Tests

Tests dedicated reranking models:
- BGE reranker models
- Model availability checking
- Automatic model pulling
- Local processing performance

```python
class TestOllamaRerankingProvider:
    async def test_dedicated_reranking(self):
        # Test Ollama dedicated reranking models
        pass
```

### HuggingFace Provider Tests

Tests local cross-encoder models:
- Cross-encoder model loading
- Batch processing efficiency
- Device configuration (CPU/GPU)
- Memory management

```python
class TestHuggingFaceRerankingProvider:
    async def test_local_cross_encoder(self):
        # Test HuggingFace local processing
        pass
```

## Integration Testing

### RAG Pipeline Integration

Tests reranking within the complete RAG workflow:

```python
async def test_rag_with_reranking():
    """Test end-to-end RAG with reranking enabled."""
    # 1. Query processing
    # 2. Similarity search
    # 3. Reranking application
    # 4. Result verification
```

### Provider Switching

Tests dynamic switching between providers:

```python
async def test_provider_switching():
    """Test switching between different reranking providers."""
    # Test switching from OpenAI to Ollama to HuggingFace
```

### Compatibility Testing

Tests compatibility with existing RAG strategies:

```python
async def test_reranking_with_contextual_embeddings():
    """Test reranking combined with contextual embeddings."""
    # Verify both features work together
```

## Error Handling Testing

### Network Failures

```python
async def test_network_timeout():
    """Test graceful handling of network timeouts."""
    # Simulate network timeout
    # Verify fallback to original results
```

### API Rate Limits

```python
async def test_rate_limit_handling():
    """Test handling of API rate limits."""
    # Simulate rate limit responses
    # Verify retry logic with backoff
```

### Malformed Responses

```python
async def test_malformed_response():
    """Test handling of malformed provider responses."""
    # Simulate invalid response formats
    # Verify graceful error handling
```

## Performance Testing

### Latency Benchmarking

```python
async def test_reranking_latency():
    """Benchmark reranking latency across providers."""
    # Measure processing time for different result set sizes
    # Compare provider performance characteristics
```

### Throughput Testing

```python
async def test_concurrent_reranking():
    """Test concurrent reranking request handling."""
    # Multiple simultaneous reranking requests
    # Verify thread safety and resource management
```

### Memory Usage

```python
def test_memory_efficiency():
    """Test memory usage during reranking operations."""
    # Monitor memory consumption
    # Verify no memory leaks
```

## Regression Testing

### Backward Compatibility

```python
async def test_backward_compatibility():
    """Test that existing functionality still works."""
    # Verify RAG works without reranking
    # Ensure metadata preservation
    # Check API interface compatibility
```

### Performance Impact

```python
async def test_performance_regression():
    """Test that reranking doesn't cause performance regression."""
    # Compare with/without reranking performance
    # Verify acceptable overhead
```

## Continuous Integration

### GitHub Actions

Add to `.github/workflows/test.yml`:

```yaml
- name: Run Reranking Tests
  run: |
    python run_reranking_tests.py --mode full
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./htmlcov_reranking/coverage.xml
```

### Test Automation

```bash
# Pre-commit hook
#!/bin/bash
python run_reranking_tests.py --mode unit
if [ $? -ne 0 ]; then
    echo "Reranking unit tests failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=src
   pip install -e .
   ```

2. **Missing Dependencies**
   ```bash
   pip install pytest pytest-asyncio pytest-cov
   ```

3. **Provider Connection Issues**
   ```bash
   export USE_MOCK_PROVIDERS=true  # Use mocked providers
   ```

4. **Timeout Issues**
   ```bash
   export RERANKING_TIMEOUT=60  # Increase timeout
   ```

### Debug Mode

```bash
# Run with verbose logging
pytest tests/test_reranking_*.py -v -s --log-cli-level=DEBUG

# Run specific failing test
pytest tests/test_reranking_providers.py::TestOpenAIRerankingProvider::test_similarity_reranking -v -s
```

## Contributing

### Adding New Tests

1. Follow existing test patterns
2. Use appropriate pytest markers
3. Include both positive and negative test cases
4. Add documentation for complex tests
5. Update this guide if needed

### Test Naming Conventions

- `test_<functionality>` - Basic functionality test
- `test_<functionality>_error_handling` - Error scenarios
- `test_<functionality>_performance` - Performance tests
- `test_<functionality>_compatibility` - Compatibility tests

### Code Coverage Goals

- Unit tests: >90% coverage
- Integration tests: >80% coverage
- Error handling: >95% coverage
- Overall reranking module: >85% coverage

## Best Practices

1. **Use Mocks** - Mock external dependencies by default
2. **Test Isolation** - Each test should be independent
3. **Clear Assertions** - Use descriptive assertion messages
4. **Performance Awareness** - Include timing assertions where relevant
5. **Documentation** - Document complex test scenarios
6. **Maintainability** - Keep tests simple and readable

## Metrics and Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
python run_reranking_tests.py --mode coverage

# View coverage report
open htmlcov_reranking/index.html
```

### Performance Metrics

The test suite tracks:
- Reranking latency per provider
- Memory usage during operations
- Throughput under concurrent load
- Error rates and recovery times

### Test Results

Test results are available in:
- Console output (immediate feedback)
- JUnit XML format (`test-results.xml`)
- HTML coverage reports (`htmlcov_reranking/`)
- Performance benchmarks (logged output)

This comprehensive test suite ensures the reranking functionality is robust, performant, and maintains compatibility with existing systems.