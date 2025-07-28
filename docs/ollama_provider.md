# Ollama AI Provider

The Ollama AI Provider enables the use of locally-hosted Ollama models for both embeddings and text generation in the Crawl4AI RAG MCP server. This provider offers the advantage of running AI models entirely locally without sending data to external APIs.

## Features

### Core Capabilities
- **Local AI Processing**: All models run locally through Ollama
- **Hybrid Provider**: Supports both embeddings and LLM functionality
- **Automatic Model Management**: Downloads models automatically when needed
- **Health Monitoring**: Comprehensive health checks and monitoring
- **Streaming Support**: Full streaming support for LLM responses
- **Error Resilience**: Robust retry logic and error handling
- **Performance Metrics**: Detailed performance tracking and metrics

### Supported Models

#### Embedding Models
- `mxbai-embed-large` (1024 dims, 669MB) - **Default**
- `nomic-embed-text` (768 dims, 274MB)
- `all-minilm` (384 dims, 46MB)
- `snowflake-arctic-embed` (1024 dims, 669MB)
- `bge-large` (1024 dims, 1.34GB)
- `bge-base` (768 dims, 438MB)
- `gte-large` (1024 dims, 670MB)
- `gte-base` (768 dims, 219MB)
- `e5-large` (1024 dims, 1.34GB)
- `e5-base` (768 dims, 438MB)
- `multilingual-e5-large` (1024 dims, 2.24GB)

#### LLM Models
- **Llama Models**:
  - `llama3.1` (4.7GB, 131K context) - **Default**
  - `llama3.1:8b` (4.7GB, 131K context)
  - `llama3.1:70b` (40GB, 131K context)
  - `llama3` (4.7GB, 8K context)
  - `llama2` (3.8GB, 4K context)

- **Code Models**:
  - `codellama` (3.8GB, 16K context)
  - `codellama:13b` (7.3GB, 16K context)
  - `codeqwen` (4.2GB, 65K context)

- **Mistral Models**:
  - `mistral` (4.1GB, 32K context)
  - `mixtral:8x7b` (26GB, 32K context)

- **Other Models**:
  - `gemma2:9b` (5.4GB, 8K context)
  - `qwen2:7b` (4.4GB, 131K context)
  - `phi3:mini` (2.3GB, 128K context)

## Installation & Setup

### Prerequisites

1. **Install Ollama**: Visit [ollama.ai](https://ollama.ai/) and install Ollama for your platform
2. **Start Ollama Service**:
   ```bash
   ollama serve
   ```
3. **Verify Installation**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Configuration

Add the following to your `.env` file:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_LLM_MODEL=llama3.1
OLLAMA_AUTO_PULL_MODELS=true
OLLAMA_TIMEOUT=300
OLLAMA_BATCH_SIZE=10
```

## Usage Examples

### Basic Configuration

```python
from ai_providers.factory import AIProviderFactory
from ai_providers.base import AIProvider

# Configuration
config = {
    "base_url": "http://localhost:11434",
    "embedding_model": "mxbai-embed-large",
    "llm_model": "llama3.1",
    "auto_pull_models": True,
    "timeout": 300.0,
    "batch_size": 10
}

# Create provider
provider = AIProviderFactory.create_hybrid_provider(AIProvider.OLLAMA, config)
await provider.initialize()
```

### Embedding Generation

```python
# Single embedding
result = await provider.create_embedding("Hello, world!")
print(f"Dimensions: {result.dimensions}")
print(f"Embedding: {result.embedding[:5]}...")

# Batch embeddings
texts = [
    "First document content",
    "Second document content", 
    "Third document content"
]
results = await provider.create_embeddings_batch(texts)
for i, result in enumerate(results):
    print(f"Text {i}: {result.dimensions} dimensions")
```

### Text Generation

```python
# Simple generation
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

result = await provider.generate(
    messages=messages,
    temperature=0.7,
    max_tokens=500
)
print(f"Response: {result.content}")

# Streaming generation
async for chunk in provider.generate_stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.metadata and chunk.metadata.get("is_final"):
        print("\n[Generation complete]")
        break
```

### Model Management

```python
# List available models
models = await provider.list_available_models()
print("Embedding models:", models["embedding_models"])
print("LLM models:", models["llm_models"])

# Pull a specific model
await provider.pull_model("llama3.1:8b")

# Health check
health = await provider.health_check()
print(f"Healthy: {health.is_healthy}")
print(f"Response time: {health.response_time_ms}ms")
```

## Configuration Reference

### Core Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | string | `http://localhost:11434` | Ollama service URL |
| `embedding_model` | string | `mxbai-embed-large` | Default embedding model |
| `llm_model` | string | `llama3.1` | Default LLM model |
| `timeout` | float | `300.0` | Request timeout (seconds) |
| `connect_timeout` | float | `10.0` | Connection timeout (seconds) |

### Model Management

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_pull_models` | bool | `true` | Automatically download missing models |
| `pull_timeout` | float | `600.0` | Model download timeout (seconds) |

### Performance Tuning

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `10` | Embedding batch size |
| `max_retries` | int | `3` | Maximum retry attempts |
| `initial_retry_delay` | float | `2.0` | Initial retry delay (seconds) |
| `max_retry_delay` | float | `60.0` | Maximum retry delay (seconds) |
| `retry_exponential_base` | float | `2.0` | Retry backoff multiplier |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `0.7` | Generation randomness (0.0-1.0) |
| `max_tokens` | int | `1000` | Maximum tokens to generate |

## Performance Considerations

### Model Selection

- **For Embeddings**:
  - Use `all-minilm` for fast, lightweight embeddings
  - Use `mxbai-embed-large` for balanced performance/quality
  - Use `bge-large` for highest quality embeddings

- **For Text Generation**:
  - Use `llama3.1` for general purpose tasks
  - Use `codellama` for code-related tasks
  - Use `phi3:mini` for resource-constrained environments

### Hardware Requirements

- **Minimum**: 8GB RAM, 4GB free disk space
- **Recommended**: 16GB RAM, 50GB free disk space
- **GPU**: Optional but significantly improves performance
  - NVIDIA GPU with CUDA support recommended
  - AMD GPU with ROCm support (limited)

### Performance Tips

1. **Model Size**: Smaller models = faster inference, larger models = better quality
2. **Batch Processing**: Use batch embedding for multiple texts
3. **Context Length**: Shorter contexts = faster generation
4. **Temperature**: Lower temperature = more deterministic output
5. **Caching**: Keep models loaded between requests

## Error Handling

The provider includes comprehensive error handling:

### Connection Errors
- Automatic retry with exponential backoff
- Health check validation
- Service availability monitoring

### Model Errors
- Automatic model downloading
- Model availability validation
- Graceful fallback for missing models

### Generation Errors
- Retry logic for temporary failures
- Proper error propagation
- Detailed error logging

## Monitoring & Metrics

### Available Metrics

```python
metrics = provider.get_metrics()
print("Embedding metrics:", metrics["embedding"])
print("LLM metrics:", metrics["llm"]) 
print("Combined metrics:", metrics["combined"])
```

### Metrics Include
- Request counts and success rates
- Token usage (estimated)
- Error rates and types
- Response times
- Model pull statistics

### Health Monitoring

```python
health = await provider.health_check()
print(f"Service healthy: {health.is_healthy}")
print(f"Response time: {health.response_time_ms}ms")
print(f"Available models: {health.metadata['available_models']}")
```

## Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Ensure Ollama is running: `ollama serve`
   - Check port 11434 is accessible
   - Verify firewall settings

2. **"Model not found"**
   - Enable auto-pulling: `auto_pull_models: true`
   - Manually pull model: `ollama pull <model-name>`
   - Check model name spelling

3. **Slow performance**
   - Use smaller models for testing
   - Enable GPU acceleration if available
   - Reduce batch sizes or context length

4. **Out of memory**
   - Use smaller models (e.g., `phi3:mini`)
   - Reduce concurrent requests
   - Increase system swap space

### Debug Logging

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("ai_providers.providers.ollama_provider").setLevel(logging.DEBUG)
```

## Advanced Usage

### Custom Model Configuration

```python
# Add custom model with dimensions
provider.embedding_provider.MODEL_CONFIGS["custom-embed"] = {
    "dimensions": 768,
    "size": "500MB",
    "family": "bert"
}

# Use custom model
result = await provider.create_embedding("test", model="custom-embed")
```

### Multi-Model Setup

```python
# Different models for different tasks
embedding_config = {
    "base_url": "http://localhost:11434",
    "embedding_model": "mxbai-embed-large"
}

llm_config = {
    "base_url": "http://localhost:11434", 
    "llm_model": "codellama:13b"
}

# Create specialized providers
embedding_provider = AIProviderFactory.create_embedding_provider(
    AIProvider.OLLAMA, embedding_config
)
llm_provider = AIProviderFactory.create_llm_provider(
    AIProvider.OLLAMA, llm_config
)
```

### Integration with RAG Pipeline

```python
# Example RAG integration
class OllamaRAGPipeline:
    def __init__(self):
        self.provider = AIProviderFactory.create_hybrid_provider(
            AIProvider.OLLAMA,
            {
                "embedding_model": "mxbai-embed-large",
                "llm_model": "llama3.1",
                "auto_pull_models": True
            }
        )
    
    async def embed_documents(self, documents):
        return await self.provider.create_embeddings_batch(documents)
    
    async def generate_response(self, query, context):
        messages = [
            {"role": "system", "content": "Use the provided context to answer questions."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        return await self.provider.generate(messages)
```

## Security Considerations

### Local Processing Benefits
- No data sent to external APIs
- Complete control over model and data
- Compliance with data privacy regulations
- Air-gapped deployment possible

### Network Security
- Ollama runs on localhost by default
- Can be configured for internal network only
- SSL/TLS support for remote deployments
- Authentication available in Ollama Pro

### Model Integrity
- Models downloaded from official Ollama registry
- Checksum verification during download
- Local model storage and management

## License & Support

The Ollama provider is part of the Crawl4AI MCP project and follows the same licensing terms. For Ollama-specific issues:

- **Ollama Documentation**: [ollama.ai/docs](https://ollama.ai/docs)
- **Ollama GitHub**: [github.com/ollama/ollama](https://github.com/ollama/ollama)
- **Model Library**: [ollama.ai/library](https://ollama.ai/library)

For provider-specific issues, please refer to the main project documentation and issue tracker.