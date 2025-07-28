# [STORY-3] Provider Configuration, Migration, and Testing

## Story Overview
**As a** system administrator and developer  
**I want** comprehensive configuration management, migration support, and testing for Ollama integration  
**So that** I can safely deploy and maintain the system with confidence in both OpenAI and Ollama providers

## Acceptance Criteria

### AC1: Environment Configuration Management
- [ ] Complete environment variable documentation with examples
- [ ] Configuration validation on startup with clear error messages
- [ ] Default configurations that work out-of-the-box
- [ ] Environment templates for different deployment scenarios
- [ ] Configuration precedence rules clearly documented

### AC2: Provider Migration and Compatibility
- [ ] Migration scripts for changing embedding dimensions
- [ ] Data backup recommendations before provider switches
- [ ] Compatibility checker utility for existing databases
- [ ] Migration guides for different scenarios (OpenAI→Ollama, dimension changes)
- [ ] Rollback procedures documented and tested

### AC3: Docker and Deployment Integration
- [ ] Docker Compose profiles for Ollama deployment
- [ ] Ollama service integration in containers
- [ ] Multi-stage builds with provider-specific optimizations
- [ ] Health checks for both OpenAI and Ollama providers
- [ ] Volume mounting for Ollama models and configurations

### AC4: Comprehensive Testing Framework
- [ ] Extended pytest suite covering all provider combinations
- [ ] Integration tests with real Ollama instances
- [ ] Performance benchmarking for provider comparison
- [ ] Load testing for local vs. remote processing
- [ ] Dimension compatibility testing across all scenarios

### AC5: Documentation and User Experience
- [ ] Complete setup guides for Ollama installation
- [ ] Model recommendation guide for different use cases
- [ ] Troubleshooting guide for common issues
- [ ] Performance tuning documentation
- [ ] Migration cookbook with examples

## Technical Implementation Notes

### Files to Create/Modify:

#### Configuration Files:
- `.env.ollama` - Ollama-specific environment template
- `.env.openai` - OpenAI-specific environment template  
- `.env.hybrid` - Mixed provider environment template
- `docker-compose.ollama.yml` - Ollama deployment profile

#### Scripts and Utilities:
- `scripts/migrate_embeddings.py` - Embedding dimension migration utility
- `scripts/validate_config.py` - Configuration validation script
- `scripts/benchmark_providers.py` - Performance comparison utility
- `scripts/setup_ollama.py` - Automated Ollama setup helper

#### Testing Extensions:
- `tests/test_ollama_integration.py` - Comprehensive Ollama testing
- `tests/test_provider_switching.py` - Provider migration testing
- `tests/test_dimension_migration.py` - Dimension change testing
- `tests/benchmarks/` - Performance testing suite

#### Documentation:
- `docs/OLLAMA_SETUP.md` - Complete Ollama setup guide
- `docs/MIGRATION_GUIDE.md` - Provider migration documentation
- `docs/CONFIGURATION.md` - Environment variable reference
- `docs/TROUBLESHOOTING.md` - Common issues and solutions

### Key Implementation Details:

#### Configuration Validation
```python
class ConfigurationValidator:
    def validate_ai_provider_config(self) -> List[str]:
        """Validate AI provider configuration and return issues"""
        issues = []
        
        ai_provider = os.getenv("AI_PROVIDER", "openai")
        if ai_provider == "ollama":
            issues.extend(self._validate_ollama_config())
        elif ai_provider == "openai":
            issues.extend(self._validate_openai_config())
        
        return issues
    
    def _validate_ollama_config(self) -> List[str]:
        """Validate Ollama-specific configuration"""
        issues = []
        
        # Check Ollama endpoint accessibility
        # Validate embedding and LLM models
        # Check dimension compatibility
        # Verify model availability
        
        return issues
```

#### Migration Utility
```python
class EmbeddingMigrationTool:
    def __init__(self, old_provider: str, new_provider: str):
        self.old_provider = old_provider
        self.new_provider = new_provider
    
    async def migrate_embeddings(self, batch_size: int = 100):
        """Migrate embeddings between providers"""
        # Backup existing data
        # Re-generate embeddings with new provider
        # Validate embedding quality
        # Update database with new embeddings
        
    def estimate_migration_time(self) -> Dict[str, Any]:
        """Estimate migration time and resources"""
        # Calculate document count
        # Estimate processing time per provider
        # Return resource requirements
```

#### Docker Integration
```yaml
# docker-compose.ollama.yml
version: '3.8'
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    environment:
      - OLLAMA_MODELS=/root/.ollama/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  mcp-crawl4ai-ollama:
    build:
      context: .
      args:
        AI_PROVIDER: ollama
    depends_on:
      - ollama
    environment:
      - AI_PROVIDER=ollama
      - OLLAMA_ENDPOINT=http://ollama:11434
      - OLLAMA_EMBEDDING_MODEL=nomic-embed-text
      - OLLAMA_LLM_MODEL=llama2
    ports:
      - "8051:8051"

volumes:
  ollama_models:
```

## Testing Requirements

### Configuration Testing
- [ ] All environment variable combinations
- [ ] Configuration validation edge cases
- [ ] Startup behavior with invalid configurations
- [ ] Default value fallbacks

### Migration Testing
- [ ] OpenAI to Ollama embedding migration
- [ ] Dimension change migrations (1536→768, 768→384)
- [ ] Rollback scenarios
- [ ] Data integrity verification
- [ ] Performance impact of migrations

### Docker Testing
- [ ] Ollama service container startup
- [ ] Model downloading and initialization
- [ ] Health check reliability
- [ ] Multi-container orchestration
- [ ] Volume persistence

### Performance Testing
- [ ] Embedding generation speed comparison
- [ ] LLM processing latency comparison
- [ ] Memory usage profiling
- [ ] Concurrent request handling
- [ ] Cold start vs. warm performance

### Integration Testing
- [ ] Full end-to-end workflows with Ollama
- [ ] Provider switching without service restart
- [ ] Error recovery scenarios
- [ ] Load balancing between providers (if implemented)

## Environment Variables Reference

### Core Configuration
```bash
# AI Provider Selection
AI_PROVIDER=ollama|openai                    # Default: openai

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
MODEL_CHOICE=gpt-4o-mini                     # Default LLM model

# Ollama Configuration  
OLLAMA_ENDPOINT=http://localhost:11434       # Default
OLLAMA_EMBEDDING_MODEL=nomic-embed-text      # Required if AI_PROVIDER=ollama
OLLAMA_LLM_MODEL=llama2                      # Required if AI_PROVIDER=ollama

# Embedding Configuration
EMBEDDING_DIMENSION=1536                     # Auto-detect if not specified
EMBEDDING_BATCH_SIZE=100                     # Default batch size

# LLM Configuration
LLM_TEMPERATURE=0.1                          # Default: 0.1
LLM_MAX_TOKENS=2048                          # Default: 2048
LLM_TIMEOUT=300                              # Default: 300s
```

## Performance Expectations

### Ollama vs OpenAI Benchmarks
- **Embedding Generation**: Local processing ~2-5x slower, but no API costs
- **LLM Processing**: Local processing ~5-10x slower, depends on hardware
- **Memory Usage**: Higher for local models (~2-8GB depending on model)
- **Startup Time**: Longer for model loading (~30-60s for first request)

## Definition of Done
- [ ] All acceptance criteria met
- [ ] Complete testing suite passes
- [ ] Docker deployment works end-to-end
- [ ] Migration tools tested and documented
- [ ] Performance benchmarks completed
- [ ] Documentation comprehensive and accurate
- [ ] Code review and security review completed
- [ ] Production deployment guide validated

## Risk Mitigation
- **Risk**: Complex migration procedures causing data loss
- **Mitigation**: Comprehensive backup procedures, validation steps, rollback plans
- **Risk**: Performance degradation with Ollama
- **Mitigation**: Performance benchmarking, optimization guides, hardware recommendations
- **Risk**: Configuration complexity overwhelming users
- **Mitigation**: Clear documentation, validation tools, sensible defaults