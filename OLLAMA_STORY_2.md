# [STORY-2] Ollama LLM Provider Integration

## Story Overview
**As a** developer using the MCP Crawl4AI RAG system  
**I want** Ollama LLM support for contextual processing and content generation  
**So that** I can perform AI operations locally without external API dependencies

## Acceptance Criteria

### AC1: Ollama LLM Client Implementation
- [ ] New `OllamaLLMProvider` class implemented following existing patterns
- [ ] Ollama client connects to configurable endpoint (same as embedding provider)
- [ ] LLM model configurable via `OLLAMA_LLM_MODEL` environment variable
- [ ] Support for common Ollama models (llama2, codellama, mistral, etc.)
- [ ] Consistent chat completion interface with OpenAI provider
- [ ] Error handling with retry logic and timeout configuration

### AC2: LLM Provider Factory Integration
- [ ] Extend existing `AIProviderFactory` to include LLM providers
- [ ] Environment variable `AI_PROVIDER` controls both embedding and LLM selection
- [ ] Provider interface abstracts LLM operations
- [ ] Consistent API across OpenAI and Ollama LLM providers
- [ ] Provider-specific model validation and configuration

### AC3: Contextual Processing Integration
- [ ] Replace direct OpenAI calls with provider abstraction in:
  - Content summarization
  - Contextual embedding enhancement
  - Code example extraction and summarization
  - Knowledge graph validation (if enabled)
- [ ] Preserve existing prompt templates and formatting
- [ ] Maintain output quality and format consistency
- [ ] Support streaming responses where applicable

### AC4: Configuration and Model Management
- [ ] Automatic model availability detection from Ollama
- [ ] Model download prompts/warnings if model not available
- [ ] Temperature and other generation parameters configurable
- [ ] Token limit handling for different Ollama models
- [ ] Model-specific optimization settings

### AC5: Response Processing and Compatibility
- [ ] Response format normalization between OpenAI and Ollama
- [ ] JSON parsing for structured outputs
- [ ] Error message standardization
- [ ] Rate limiting and request queuing for local processing
- [ ] Progress indicators for long-running local LLM operations

## Technical Implementation Notes

### Files to Modify/Create:
- `src/ai/ollama_llm_provider.py` - New Ollama LLM implementation
- `src/ai/base.py` - Add LLM provider interface
- `src/ai/factory.py` - Extend for LLM provider creation
- `src/ai/openai_provider.py` - Add LLM functionality to existing provider

### Integration Points:
- `src/utils.py` - Replace direct OpenAI calls with provider abstraction
- `src/crawl4ai_mcp_refactored.py` - Update context management
- Knowledge graph modules - Replace OpenAI calls if knowledge graph enabled

### Configuration Updates:
- Add `OLLAMA_LLM_MODEL` environment variable
- Add `LLM_TEMPERATURE` environment variable (default: 0.1)
- Add `LLM_MAX_TOKENS` environment variable
- Add `LLM_TIMEOUT` environment variable (default: 300s for local processing)

### Key Implementation Details:

#### Ollama LLM Provider
```python
class OllamaLLMProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self._validate_model_availability()
    
    async def generate_completion(self, messages: List[Dict], **kwargs) -> str:
        # Chat completion with Ollama API
        # Handle streaming responses
        # Error handling and retries
        # Response format normalization
    
    async def generate_summary(self, content: str, max_length: int = 500) -> str:
        # Content summarization
        # Template-based prompting
        # Length control and validation
```

#### Provider Interface Extension
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_completion(self, messages: List[Dict], **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_summary(self, content: str, max_length: int) -> str:
        pass
    
    @abstractmethod  
    async def extract_structured_data(self, content: str, schema: Dict) -> Dict:
        pass
```

#### Utils Integration
```python
async def get_llm_provider() -> LLMProvider:
    """Get configured LLM provider (OpenAI or Ollama)"""
    ai_provider = os.getenv("AI_PROVIDER", "openai")
    return AIProviderFactory.create_llm_provider(ai_provider)

async def generate_contextual_summary(content: str) -> str:
    """Generate summary using configured LLM provider"""
    llm = await get_llm_provider()
    return await llm.generate_summary(content)
```

## Testing Requirements

### Unit Tests
- [ ] Ollama LLM provider with mocked API calls
- [ ] Response format normalization
- [ ] Error handling and retry logic
- [ ] Model validation and configuration

### Integration Tests
- [ ] Ollama LLM generation (with local Ollama instance)
- [ ] Contextual processing with Ollama models
- [ ] Provider switching between OpenAI and Ollama
- [ ] Streaming response handling

### Compatibility Tests
- [ ] Existing OpenAI LLM functionality unchanged
- [ ] Output quality and format consistency
- [ ] Performance comparison between providers
- [ ] Error handling parity

### Model-Specific Tests
- [ ] Test with different Ollama models (llama2, mistral, etc.)
- [ ] Token limit handling
- [ ] Temperature and parameter effects
- [ ] Model availability detection

## Dependencies
- `ollama` Python client library (already added in Story 1)
- Local Ollama installation with at least one LLM model
- Model-specific configuration documentation

## Performance Considerations
- Local LLM processing may be slower than OpenAI API
- Memory usage considerations for larger models
- Concurrent request handling for local processing
- Progress indicators for user experience

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Unit and integration tests pass
- [ ] Existing OpenAI LLM functionality preserved
- [ ] Documentation updated with model configuration
- [ ] Code review completed
- [ ] Performance benchmarking completed
- [ ] Local processing provides comparable output quality

## Risk Mitigation
- **Risk**: Local LLM performance significantly slower than OpenAI
- **Mitigation**: Performance benchmarking, async processing, progress indicators
- **Risk**: Different output formats breaking existing functionality
- **Mitigation**: Response normalization, extensive compatibility testing
- **Risk**: Model availability issues in different environments
- **Mitigation**: Model detection, clear error messages, fallback suggestions