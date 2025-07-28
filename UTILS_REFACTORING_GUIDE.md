# Utils Refactoring Guide: AI Provider Abstraction

This document provides a comprehensive guide for the refactored `utils.py` that integrates with the AI provider abstraction system.

## Overview

The refactored `utils_refactored.py` maintains full backward compatibility with the original `utils.py` while adding support for multiple AI providers through an abstraction layer. This allows seamless switching between OpenAI, Ollama, and other AI providers without code changes.

## Key Features

### 1. **Backward Compatibility**
- All existing function signatures remain unchanged
- Drop-in replacement for `utils.py`
- No breaking changes to existing code

### 2. **AI Provider Abstraction**
- Support for multiple AI providers (OpenAI, Ollama, Anthropic, Cohere, HuggingFace)
- Automatic provider selection based on environment variables
- Provider health checks and fallback logic
- Configuration validation on startup

### 3. **Enhanced Error Handling**
- Graceful fallback to zero embeddings on failure
- Individual embedding fallback when batch processing fails
- Retry logic with exponential backoff
- Provider reconnection on connection failures

### 4. **Performance Improvements**
- Async/await pattern for better concurrency
- Batch processing optimization
- Connection pooling and reuse
- Memory-efficient processing of large documents

### 5. **Monitoring and Observability**
- Provider health monitoring
- Performance metrics collection
- Request/response tracking
- Error rate monitoring

## Migration Guide

### Step 1: Environment Configuration

Configure your AI provider by setting environment variables:

#### OpenAI (Default)
```bash
export AI_PROVIDER=openai
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_LLM_MODEL=gpt-4o-mini
export OPENAI_EMBEDDING_DIMENSIONS=1536
```

#### Ollama
```bash
export AI_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_LLM_MODEL=llama3.2
export OLLAMA_EMBEDDING_DIMENSIONS=768
```

#### Anthropic (LLM only)
```bash
export AI_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
export ANTHROPIC_LLM_MODEL=claude-3-sonnet-20240229
# Note: You'll need a separate embedding provider
export OPENAI_API_KEY=your_openai_key  # For embeddings
```

### Step 2: Code Migration

Replace imports in your code:

```python
# Old import
from utils import create_embedding, create_embeddings_batch

# New import
from utils_refactored import create_embedding, create_embeddings_batch
```

**That's it!** No other code changes are required.

### Step 3: Validation

Validate your configuration:

```python
from utils_refactored import validate_ai_provider_config, get_ai_provider_info

# Check configuration
config_status = validate_ai_provider_config()
print("Configuration valid:", config_status["valid"])

# Get provider information
provider_info = get_ai_provider_info()
print("Provider:", provider_info["provider"])
print("Supports embeddings:", provider_info["supports_embeddings"])
print("Supports LLM:", provider_info["supports_llm"])
```

## Advanced Usage

### Provider Management

```python
from utils_refactored import (
    initialize_ai_providers,
    health_check_ai_providers,
    get_ai_provider_metrics,
    close_ai_providers
)

# Initialize providers explicitly
await initialize_ai_providers()

# Check provider health
health_status = await health_check_ai_providers()
print("Embedding provider healthy:", health_status["embedding"]["healthy"])

# Get performance metrics
metrics = await get_ai_provider_metrics()
print("Request count:", metrics["embedding"]["request_count"])
print("Error rate:", metrics["embedding"]["error_rate"])

# Clean shutdown
await close_ai_providers()
```

### Context Manager Usage

For advanced usage with explicit lifecycle management:

```python
from utils_refactored import ai_provider_context

async def process_documents():
    async with ai_provider_context() as provider_manager:
        # Provider is automatically initialized
        embeddings = create_embeddings_batch(["text1", "text2"])
        return embeddings
    # Provider connections are maintained for reuse
```

### Configuration Validation

```python
from utils_refactored import validate_ai_provider_requirements

# Comprehensive validation
requirements = validate_ai_provider_requirements()
print("Valid configuration:", requirements["valid"])
print("Available providers:", requirements["available"])
print("Provider capabilities:", requirements["capabilities"])
```

## Function Reference

### Core Functions (Backward Compatible)

#### `create_embedding(text: str) -> List[float]`
Creates an embedding for a single text.

**Parameters:**
- `text`: Text to create embedding for

**Returns:**
- List of floats representing the embedding

**Example:**
```python
embedding = create_embedding("Hello world")
print(f"Embedding dimensions: {len(embedding)}")
```

#### `create_embeddings_batch(texts: List[str]) -> List[List[float]]`
Creates embeddings for multiple texts efficiently.

**Parameters:**
- `texts`: List of texts to create embeddings for

**Returns:**
- List of embeddings (each embedding is a list of floats)

**Example:**
```python
texts = ["Hello", "world", "AI", "embeddings"]
embeddings = create_embeddings_batch(texts)
print(f"Created {len(embeddings)} embeddings")
```

#### `generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]`
Generates contextual information for better retrieval.

**Parameters:**
- `full_document`: Complete document text
- `chunk`: Specific chunk to contextualize

**Returns:**
- Tuple of (contextual_text, success_flag)

**Example:**
```python
full_doc = "This is a comprehensive guide about AI..."
chunk = "AI embeddings are vector representations..."
contextual_text, success = generate_contextual_embedding(full_doc, chunk)
```

### New Management Functions

#### `initialize_ai_providers(force_reinit: bool = False) -> None`
Explicitly initializes AI providers.

#### `close_ai_providers() -> None`
Closes all AI provider connections.

#### `health_check_ai_providers() -> Dict[str, Any]`
Returns health status of all providers.

#### `get_ai_provider_info() -> Dict[str, Any]`
Returns information about configured providers.

#### `validate_ai_provider_config() -> Dict[str, Any]`
Validates provider configuration.

#### `get_available_ai_providers() -> Dict[str, List[str]]`
Lists all available AI providers.

#### `get_ai_provider_metrics() -> Dict[str, Any]`
Returns performance metrics.

## Configuration Options

### Provider-Specific Settings

#### OpenAI Configuration
```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_ORGANIZATION=your_org_id  # Optional
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_DIMENSIONS=1536
OPENAI_MAX_BATCH_SIZE=100
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=30.0
```

#### Ollama Configuration
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_DIMENSIONS=768
OLLAMA_MAX_BATCH_SIZE=10
OLLAMA_MAX_RETRIES=3
OLLAMA_TIMEOUT=120.0
OLLAMA_PULL_MODELS=false  # Auto-pull missing models
```

#### Anthropic Configuration
```bash
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_BASE_URL=https://api.anthropic.com  # Optional
ANTHROPIC_LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_RETRIES=3
ANTHROPIC_TIMEOUT=60.0
```

### Global Settings
```bash
AI_PROVIDER=openai  # Default provider
USE_CONTEXTUAL_EMBEDDINGS=true  # Enable contextual embeddings
MODEL_CHOICE=gpt-4o-mini  # Backward compatibility
```

## Error Handling

The refactored utils provides comprehensive error handling:

### 1. **Provider Connection Failures**
- Automatic retry with exponential backoff
- Health check validation
- Graceful degradation to fallback providers

### 2. **Embedding Generation Failures**
- Batch processing fallback to individual requests
- Zero embedding fallback on complete failure
- Dimension validation and consistency checks

### 3. **Configuration Errors**
- Startup validation of all required settings
- Clear error messages for missing configurations
- Environment variable validation

## Performance Considerations

### 1. **Batch Processing**
- Always use `create_embeddings_batch()` for multiple texts
- Automatic batching optimization based on provider limits
- Concurrent processing where supported

### 2. **Connection Reuse**
- Provider connections are maintained across requests
- Connection pooling for improved performance
- Automatic reconnection on failures

### 3. **Memory Management**
- Efficient processing of large documents
- Streaming support where available
- Garbage collection optimization

## Monitoring and Debugging

### Health Checks
```python
health_status = await health_check_ai_providers()
if not health_status["embedding"]["healthy"]:
    print(f"Embedding provider error: {health_status['embedding']['error']}")
```

### Performance Metrics
```python
metrics = await get_ai_provider_metrics()
print(f"Total requests: {metrics['embedding']['request_count']}")
print(f"Error rate: {metrics['embedding']['error_rate']:.2%}")
print(f"Average response time: {metrics['embedding']['avg_response_time_ms']:.1f}ms")
```

### Debug Logging
Enable debug logging for detailed information:
```python
import logging
logging.getLogger('utils_refactored').setLevel(logging.DEBUG)
logging.getLogger('ai_providers').setLevel(logging.DEBUG)
```

## Testing

### Unit Tests
Run the comprehensive test suite:
```bash
python -m pytest tests/test_utils_refactored.py -v
```

### Integration Tests
Test with actual providers:
```bash
# Test OpenAI integration
export OPENAI_API_KEY=your_key
python -m pytest tests/test_utils_refactored.py::TestAIProviderIntegration -v

# Test Ollama integration (requires Ollama running)
export AI_PROVIDER=ollama
python -m pytest tests/test_utils_refactored.py::TestAIProviderIntegration -v
```

### Performance Tests
```bash
python -m pytest tests/test_utils_refactored.py::TestPerformanceAndConcurrency -v
```

## Troubleshooting

### Common Issues

#### 1. **Provider Not Available**
```
Error: Unsupported AI provider: invalid_provider
```
**Solution:** Check available providers:
```python
from utils_refactored import get_available_ai_providers
print(get_available_ai_providers())
```

#### 2. **Configuration Missing**
```
Error: Missing required configuration for openai: ['api_key']
```
**Solution:** Set required environment variables:
```bash
export OPENAI_API_KEY=your_api_key
```

#### 3. **Dimension Mismatch**
```
Error: Embedding dimension mismatch: expected 1536, got 768
```
**Solution:** Update dimension configuration:
```bash
export OPENAI_EMBEDDING_DIMENSIONS=768  # Match your model
```

#### 4. **Connection Timeout**
```
Error: Provider connection failed: timeout
```
**Solution:** Increase timeout or check provider availability:
```bash
export OPENAI_TIMEOUT=60.0  # Increase timeout
```

### Debug Steps

1. **Validate Configuration**
   ```python
   from utils_refactored import validate_ai_provider_config
   print(validate_ai_provider_config())
   ```

2. **Check Provider Health**
   ```python
   from utils_refactored import health_check_ai_providers
   print(await health_check_ai_providers())
   ```

3. **Test Basic Functionality**
   ```python
   from utils_refactored import create_embedding
   try:
       embedding = create_embedding("test")
       print(f"Success: {len(embedding)} dimensions")
   except Exception as e:
       print(f"Error: {e}")
   ```

## Migration Checklist

- [ ] Set up environment variables for your chosen AI provider
- [ ] Validate configuration using `validate_ai_provider_config()`
- [ ] Update imports in your code from `utils` to `utils_refactored`
- [ ] Test basic functionality with `create_embedding("test")`
- [ ] Run health checks to ensure provider connectivity
- [ ] Update deployment scripts with new environment variables
- [ ] Run integration tests to verify functionality
- [ ] Monitor performance metrics after deployment
- [ ] Set up monitoring for provider health status
- [ ] Document provider-specific configurations for your team

## Best Practices

1. **Provider Selection**
   - Use OpenAI for production workloads requiring high accuracy
   - Use Ollama for local development and privacy-sensitive applications
   - Consider costs and rate limits when choosing providers

2. **Configuration Management**
   - Use environment variables for all provider settings
   - Validate configuration at startup
   - Implement graceful degradation for provider failures

3. **Error Handling**
   - Always check provider health before critical operations
   - Implement retry logic for transient failures
   - Log errors with sufficient context for debugging

4. **Performance Optimization**
   - Use batch processing for multiple embeddings
   - Monitor provider metrics and adjust batch sizes
   - Implement caching for frequently requested embeddings

5. **Security**
   - Store API keys securely (environment variables, secrets management)
   - Use HTTPS for all provider communications
   - Implement rate limiting to avoid quota exhaustion

## Support

For questions, issues, or contributions:

1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Check provider-specific documentation
4. Create an issue with detailed error logs and configuration

## Changelog

### v1.0.0 - Initial Release
- Full backward compatibility with original utils.py
- AI provider abstraction integration
- Support for OpenAI, Ollama, Anthropic, Cohere, HuggingFace
- Comprehensive error handling and fallback logic
- Performance optimizations and monitoring
- Extensive test coverage