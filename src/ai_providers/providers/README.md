# OpenAI Provider Implementation

This document describes the OpenAI provider implementation for the AI providers system.

## Overview

The OpenAI provider implements three classes that provide comprehensive OpenAI API integration:

- **`OpenAIEmbeddingProvider`** - Handles text embeddings using OpenAI's embedding models
- **`OpenAILLMProvider`** - Handles text generation using OpenAI's chat completion models  
- **`OpenAIProvider`** - Hybrid provider that combines both embedding and LLM capabilities

## Features

### Production-Ready Features

✅ **Robust Error Handling**
- Exponential backoff retry logic with configurable parameters
- Differentiation between retryable and non-retryable errors
- Individual fallback processing for batch operations
- Comprehensive error logging and metrics

✅ **Performance Optimizations**
- Batch embedding processing (up to 50 texts per batch by default)
- Async/await throughout for non-blocking operations
- Connection reuse and proper resource management
- Configurable timeouts and rate limiting

✅ **Health Checks and Monitoring**
- Real-time health status monitoring
- Performance metrics tracking (requests, tokens, errors)
- Response time measurement
- Request/error rate monitoring

✅ **Comprehensive Model Support**
- All current OpenAI embedding models (3-small, 3-large, ada-002)
- All current OpenAI chat models (GPT-4, GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo)
- Model-specific dimension and context length configurations
- Dynamic model validation and fallbacks

✅ **Enterprise Configuration**
- Environment variable and config-based setup
- Configurable retry policies, timeouts, and batch sizes
- Support for all OpenAI API parameters
- Backwards compatibility with existing environment variables

## Configuration

### Environment Variables (Backwards Compatible)

```bash
OPENAI_API_KEY=your_api_key_here
MODEL_CHOICE=gpt-4o-mini  # For LLM operations
```

### Provider Configuration

```python
config = {
    # Authentication
    "api_key": "your_api_key",  # Required
    
    # Models
    "embedding_model": "text-embedding-3-small",  # Default
    "llm_model": "gpt-4o-mini",  # Default
    
    # Generation settings
    "temperature": 0.7,  # Default
    "max_tokens": 1000,  # Default
    
    # Performance tuning
    "batch_size": 50,  # Embedding batch size
    "timeout": 60.0,  # Request timeout in seconds
    
    # Retry configuration
    "max_retries": 3,
    "initial_retry_delay": 1.0,
    "max_retry_delay": 60.0,
    "retry_exponential_base": 2.0
}
```

## Usage Examples

### Embedding Provider

```python
from ai_providers.providers import OpenAIEmbeddingProvider

config = {"api_key": "your_key"}
provider = OpenAIEmbeddingProvider(config)

await provider.initialize()

# Single embedding
result = await provider.create_embedding("Hello world")
print(f"Dimensions: {result.dimensions}")

# Batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
results = await provider.create_embeddings_batch(texts)

# Health check
health = await provider.health_check()
print(f"Healthy: {health.is_healthy}")

await provider.close()
```

### LLM Provider

```python
from ai_providers.providers import OpenAILLMProvider

config = {"api_key": "your_key", "llm_model": "gpt-4o-mini"}
provider = OpenAILLMProvider(config)

await provider.initialize()

# Text generation
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]
result = await provider.generate(messages)
print(result.content)

# Streaming generation
async for chunk in provider.generate_stream(messages):
    if chunk.metadata.get("is_partial"):
        print(chunk.content, end="")

await provider.close()
```

### Hybrid Provider

```python
from ai_providers.providers import OpenAIProvider

config = {"api_key": "your_key"}
provider = OpenAIProvider(config)

await provider.initialize()

# Use both capabilities
embedding = await provider.create_embedding("Test text")
llm_result = await provider.generate([
    {"role": "user", "content": "Explain embeddings"}
])

# Get metrics for both
metrics = provider.get_metrics()
print(f"Embedding requests: {metrics['embedding']['request_count']}")
print(f"LLM requests: {metrics['llm']['request_count']}")

await provider.close()
```

### Factory Usage

```python
from ai_providers.base import AIProvider
from ai_providers.factory import AIProviderFactory

# Create through factory
config = {"api_key": "your_key"}
provider = AIProviderFactory.create_provider(AIProvider.OPENAI, config)

await provider.initialize()
# Provider is automatically a hybrid provider if available
await provider.close()
```

## Model Support

### Embedding Models

| Model | Dimensions | Max Tokens | Use Case |
|-------|------------|------------|----------|
| `text-embedding-3-small` | 1536 | 8191 | General purpose, cost-effective |
| `text-embedding-3-large` | 3072 | 8191 | Higher quality, more expensive |
| `text-embedding-ada-002` | 1536 | 8191 | Legacy model, still supported |

### LLM Models

| Model | Max Tokens | Context Length | Use Case |
|-------|------------|----------------|----------|
| `gpt-4o-mini` | 16384 | 128000 | Fast, cost-effective |
| `gpt-4o` | 4096 | 128000 | Latest, highest quality |
| `gpt-4-turbo` | 4096 | 128000 | Balanced performance |
| `gpt-4` | 8192 | 8192 | Original GPT-4 |
| `gpt-3.5-turbo` | 4096 | 16385 | Fast, inexpensive |

## Error Handling

The provider handles various error conditions:

### Retryable Errors
- Rate limiting (429 errors)
- Temporary server errors (500, 502, 503, 504)
- Network/connection timeouts
- OpenAI-specific temporary errors

### Retry Strategy
- Exponential backoff with jitter
- Configurable max retries (default: 3)
- Individual fallback for batch operations
- Zero-embedding fallback for failed texts

### Non-Retryable Errors
- Authentication errors (401)
- Invalid model errors (404)
- Malformed request errors (400)
- Quota exceeded errors

## Performance Characteristics

### Batch Processing
- Default batch size: 50 embeddings per request
- Automatic batching for large text lists
- Parallel processing with ThreadPoolExecutor for contextual embeddings
- Memory-efficient streaming for large datasets

### Rate Limiting
- Built-in retry with exponential backoff
- Respects OpenAI rate limits automatically
- Configurable delay parameters
- Request queuing and throttling

### Resource Management
- Proper async context management
- Connection pooling and reuse
- Memory cleanup on provider close
- Timeout handling for long-running requests

## Monitoring and Metrics

### Available Metrics
```python
metrics = provider.get_metrics()
# Returns:
{
    "request_count": 150,
    "total_tokens": 75000,  # For embeddings
    "total_input_tokens": 25000,  # For LLM
    "total_output_tokens": 5000,  # For LLM
    "error_count": 2,
    "error_rate": 0.013,
    "model": "text-embedding-3-small"
}
```

### Health Check Response
```python
health = await provider.health_check()
# Returns HealthStatus with:
# - is_healthy: bool
# - response_time_ms: float
# - error_message: Optional[str]
# - metadata: Dict with usage info
```

## Integration with Existing Code

The provider is designed to be backwards compatible with the existing `utils.py` functions:

### Migration Path

1. **Replace `create_embedding()`**:
   ```python
   # Old
   embedding = create_embedding(text)
   
   # New
   result = await provider.create_embedding(text)
   embedding = result.embedding
   ```

2. **Replace `create_embeddings_batch()`**:
   ```python
   # Old
   embeddings = create_embeddings_batch(texts)
   
   # New
   results = await provider.create_embeddings_batch(texts)
   embeddings = [r.embedding for r in results]
   ```

3. **Replace OpenAI chat completions**:
   ```python
   # Old
   response = openai.chat.completions.create(...)
   
   # New
   messages = [{"role": "user", "content": prompt}]
   result = await provider.generate(messages)
   content = result.content
   ```

## Testing

Run the test suite:

```bash
# Set your API key
export OPENAI_API_KEY=your_key_here

# Run tests
python3 test_openai_provider.py
```

The test suite covers:
- Individual provider functionality
- Batch operations and error handling
- Health checks and metrics
- Factory integration
- Performance characteristics

## Dependencies

- `openai>=1.0.0` - Official OpenAI Python client
- `asyncio` - Async/await support
- Standard library modules for logging, time, typing

## Thread Safety

The provider is designed for async/await usage and is not thread-safe for synchronous access. Use proper async context management:

```python
# Good
async with provider:
    result = await provider.create_embedding(text)

# Also good
await provider.initialize()
try:
    result = await provider.create_embedding(text)
finally:
    await provider.close()
```