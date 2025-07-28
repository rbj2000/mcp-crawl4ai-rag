# Ollama AI Provider Support Epic - Testing Framework

This comprehensive testing framework validates the implementation of Ollama AI provider support across all aspects of the MCP Crawl4AI RAG server, ensuring zero regression and production-ready quality.

## 🎯 Testing Objectives

The testing framework ensures:
- **Zero regression** in existing OpenAI functionality
- **Safe provider switching** without data corruption
- **Proper dimension validation** and migration guidance
- **Production-ready deployment** scenarios with Docker orchestration
- **Complete epic story validation** against acceptance criteria

## 📊 Test Suite Architecture

### Test Categories

1. **Unit Testing** (`test_ai_providers.py`)
   - AI provider implementations (OpenAI & Ollama)
   - Configuration validation
   - Dimension compatibility checking
   - Error handling and fallback scenarios

2. **Integration Testing** (`test_ollama_integration.py`)
   - Real Ollama server connectivity
   - Actual embedding generation with various models
   - Provider switching with dimension validation
   - Database integration scenarios

3. **Performance Testing** (`test_performance_comparison.py`)
   - OpenAI vs Ollama performance benchmarks
   - Latency and throughput analysis
   - Memory usage patterns
   - Cost-benefit analysis

4. **Migration Testing** (`test_migration_integrity.py`)
   - Dimension change migration scenarios
   - Data integrity validation
   - Rollback capabilities
   - Migration guidance workflows

5. **Docker Testing** (`test_docker_orchestration.py`)
   - Multi-container deployment scenarios
   - Health checks and service discovery
   - Network connectivity and security
   - Container scaling and failover

6. **Epic Story Validation** (`test_epic_stories.py`)
   - Story-specific acceptance criteria verification
   - End-to-end workflow validation
   - Regression coverage across all stories

## 🚀 Quick Start

### Prerequisites

```bash
# Install test dependencies
pip install -e ".[testing]"

# Optional: Install specific provider dependencies
pip install -e ".[supabase]"  # For Supabase tests
pip install -e ".[pinecone]"  # For Pinecone tests
```

### Basic Test Execution

```bash
# Run all unit tests (default - no external dependencies)
python tests/test_runner.py --suites unit

# Run integration tests (requires Ollama server)
python tests/test_runner.py --suites integration

# Run complete epic validation
python tests/test_runner.py --suites stories

# Run all test suites
python tests/test_runner.py --suites all
```

### Advanced Configuration

```bash
# Run with custom configuration
python tests/test_runner.py --config tests/config.yaml --suites all

# Enable parallel execution
python tests/test_runner.py --parallel --suites unit migration

# Fail fast on first error
python tests/test_runner.py --fail-fast --suites all
```

## 🔧 Configuration

### Environment Variables

#### Core Testing
```bash
# Test execution control
export ENABLE_UNIT_TESTS=true
export ENABLE_INTEGRATION_TESTS=false  # Requires Ollama
export ENABLE_PERFORMANCE_TESTS=false  # Long running
export ENABLE_MIGRATION_TESTS=true
export ENABLE_DOCKER_TESTS=false       # Requires Docker
export ENABLE_STORY_VALIDATION=true

# Test timeout (seconds)
export TEST_TIMEOUT=300
```

#### Provider Configuration
```bash
# OpenAI (for comparison tests)
export OPENAI_API_KEY=your_openai_api_key

# Ollama (for integration tests)
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=nomic-embed-text
export OLLAMA_LLM_MODEL=llama3.2

# Skip tests if services unavailable
export SKIP_OLLAMA_TESTS=false
export SKIP_DOCKER_TESTS=false
```

#### Database Testing
```bash
# Optional database providers for integration
export TEST_SUPABASE_URL=your_supabase_url
export TEST_SUPABASE_SERVICE_KEY=your_service_key
export TEST_PINECONE_API_KEY=your_pinecone_key
export TEST_NEO4J_URI=bolt://localhost:7687
```

### Test Configuration File

Create `tests/config.yaml` for advanced configuration:

```yaml
test_suites:
  unit:
    name: "Unit Tests"
    files: ["test_ai_providers.py"]
    markers: ["unit"]
    timeout: 300
    parallel: true
    
  integration:
    name: "Integration Tests"
    files: ["test_ollama_integration.py"]
    markers: ["integration"]
    timeout: 900
    parallel: false
    required_services: ["ollama"]

reporting:
  formats: ["console", "json", "html"]
  output_dir: "test_reports"
  coverage_threshold: 80.0

ci_integration:
  fail_fast: false
  retry_failed: true
  max_retries: 2
```

## 📋 Test Categories Detail

### Unit Tests (`test_ai_providers.py`)

Tests core AI provider functionality without external dependencies:

- **Dimension Validation**: Ensures embedding dimensions are validated
- **Provider Factory**: Tests provider creation and configuration
- **Error Handling**: Validates retry mechanisms and fallback scenarios
- **Batch Processing**: Tests concurrent and batch operations
- **Mock Scenarios**: Comprehensive testing with mock providers

**Key Test Classes:**
- `TestEmbeddingDimensions`: Dimension compatibility validation
- `TestAIProviderFactory`: Provider creation and configuration
- `TestOpenAIProvider`: OpenAI-specific functionality
- `TestOllamaProvider`: Ollama-specific functionality
- `TestProviderSwitching`: Safe switching between providers
- `TestErrorHandling`: Comprehensive error scenarios

### Integration Tests (`test_ollama_integration.py`)

Tests with real Ollama instances and external services:

- **Server Connectivity**: Ollama server health and model availability
- **Real Embeddings**: Actual embedding generation and validation
- **LLM Generation**: Text generation capabilities
- **Database Integration**: Real database operations with variable dimensions
- **Performance Metrics**: Actual latency and throughput measurements

**Key Test Classes:**
- `TestOllamaConnectivity`: Basic server connectivity
- `TestOllamaEmbeddings`: Embedding generation validation
- `TestOllamaLLM`: Text generation capabilities
- `TestProviderSwitching`: Real provider switching scenarios
- `TestOllamaPerformance`: Performance characteristics
- `TestOllamaErrorHandling`: Real error scenarios

### Performance Tests (`test_performance_comparison.py`)

Comprehensive performance analysis and benchmarking:

- **Latency Comparison**: OpenAI vs Ollama response times
- **Throughput Analysis**: Batch processing capabilities
- **Memory Usage**: Resource consumption patterns
- **Quality Metrics**: Embedding consistency and separability
- **Cost Analysis**: Performance vs cost trade-offs

**Key Test Classes:**
- `TestEmbeddingPerformance`: Embedding generation benchmarks
- `TestLLMPerformance`: Text generation benchmarks
- `TestMemoryUsage`: Memory consumption analysis
- `TestQualityMetrics`: Output quality validation
- `TestCostAnalysis`: Cost-benefit analysis

### Migration Tests (`test_migration_integrity.py`)

Validates safe migration between providers and dimensions:

- **Dimension Compatibility**: Validates incompatible dimension detection
- **Data Migration**: Full recompute and dimension mapping strategies
- **Data Integrity**: Comprehensive integrity validation
- **Rollback Scenarios**: Automatic and manual rollback testing
- **Migration Guidance**: Validation of migration workflows

**Key Test Classes:**
- `TestDimensionCompatibility`: Dimension validation logic
- `TestDataMigration`: Migration strategy testing
- `TestDataIntegrity`: Post-migration validation
- `TestMigrationGuidance`: Migration workflow validation
- `TestRollbackCapabilities`: Rollback scenario testing

### Docker Tests (`test_docker_orchestration.py`)

Container orchestration and deployment validation:

- **Multi-Container Deployment**: MCP server + Ollama + Database
- **Health Checks**: Service health monitoring
- **Network Security**: Inter-service communication
- **Scaling Scenarios**: Horizontal scaling and load balancing
- **Failover Testing**: Service failure and recovery

**Key Test Classes:**
- `TestBasicContainerDeployment`: Single container scenarios
- `TestMultiContainerOrchestration`: Docker Compose scenarios  
- `TestContainerHealthChecks`: Health monitoring
- `TestContainerScaling`: Scaling capabilities
- `TestContainerFailover`: Failure recovery
- `TestNetworkSecurity`: Security validation

### Epic Story Tests (`test_epic_stories.py`)

Validates each epic story against specific acceptance criteria:

#### Story 1: Configurable Embedding Dimensions & Ollama Embedding Provider
- **AC1**: Embedding dimensions configurable via environment variables
- **AC2**: Ollama provider generates embeddings with correct dimensions
- **AC3**: Dimension validation prevents incompatible embeddings
- **AC4**: Database operations handle variable dimensions
- **AC5**: Performance is acceptable for Ollama embeddings

#### Story 2: Ollama LLM Provider Integration  
- **AC1**: Ollama LLM generates coherent text responses
- **AC2**: LLM processing matches OpenAI capabilities for RAG
- **AC3**: Error handling and timeout management work correctly
- **AC4**: LLM responses are contextually appropriate

#### Story 3: Provider Configuration, Migration, and Testing
- **AC1**: Configuration allows seamless provider switching
- **AC2**: Dimension validation prevents data corruption
- **AC3**: All existing tests pass with both providers
- **AC4**: Docker support includes Ollama integration
- **AC5**: Migration guidance prevents incompatible changes

## 🏃‍♂️ Running Tests

### Local Development

```bash
# Quick validation (unit tests only)
python tests/test_runner.py --suites unit

# Full validation (requires all services)
python tests/test_runner.py --suites all

# Story-specific validation
python tests/test_runner.py --suites stories
```

### CI/CD Integration

```bash
# CI pipeline (safe tests only)
python tests/test_runner.py --suites unit migration stories

# Full integration (with services)
python tests/test_runner.py --suites all --fail-fast
```

### Docker Testing

```bash
# Start required services
docker-compose up -d ollama

# Run Docker-specific tests
python tests/test_runner.py --suites docker

# Full orchestration testing
python tests/test_runner.py --suites integration docker
```

## 📊 Test Reports

The test runner generates comprehensive reports in multiple formats:

### Console Output
- Real-time test execution status
- Summary statistics and timing
- Epic validation results
- Coverage information

### JSON Report (`test_reports/test_results.json`)
```json
{
  "execution_summary": {
    "total_duration": 120.45,
    "total_tests": 156,
    "total_passed": 154,
    "total_failed": 2,
    "overall_success": false
  },
  "suite_results": [...]
}
```

### HTML Report (`test_reports/test_results.html`)
- Interactive web-based report
- Detailed failure analysis
- Coverage visualization
- Epic story validation status

### Coverage Reports
- Terminal output with missing lines
- HTML coverage report (`test_reports/coverage_html/`)
- JSON coverage data (`test_reports/coverage.json`)

## 🔍 Test Fixtures and Utilities

### Common Fixtures (`conftest.py`)
- **Provider Configurations**: Pre-configured AI providers
- **Test Data**: Documents, code examples, prompts
- **Database Configs**: Various dimension configurations
- **Performance Data**: Benchmarking datasets
- **Docker Configs**: Container orchestration templates

### Utility Functions
- **Cosine Similarity**: Vector similarity calculations
- **Mock Embedding Generation**: Deterministic test embeddings
- **Quality Validation**: Embedding quality assessment
- **Database Configuration**: Dynamic test database setup

## 🐛 Debugging and Troubleshooting

### Common Issues

#### Ollama Tests Failing
```bash
# Check Ollama server status
curl http://localhost:11434/api/tags

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2

# Skip Ollama tests if unavailable
export SKIP_OLLAMA_TESTS=true
```

#### Docker Tests Failing
```bash
# Check Docker availability
docker version

# Skip Docker tests if unavailable
export SKIP_DOCKER_TESTS=true

# Run with Docker Compose
docker-compose -f tests/docker-compose.test.yml up -d
```

#### Performance Tests Slow
```bash
# Disable performance tests in CI
export ENABLE_PERFORMANCE_TESTS=false

# Run performance tests separately
python tests/test_runner.py --suites performance
```

### Verbose Debugging

```bash
# Enable verbose output
python tests/test_runner.py --suites unit -v

# Run specific test class
python -m pytest tests/test_ai_providers.py::TestEmbeddingDimensions -v

# Enable debug logging
export PYTEST_LOGGING_LEVEL=DEBUG
```

## 🎯 Validation Criteria

### Epic Success Criteria

The epic is considered successful when:

1. **All Story Acceptance Criteria Pass** (100% pass rate required)
2. **Zero Regression** in existing functionality
3. **Performance Acceptable** (within 2x of baseline)
4. **Docker Integration** working correctly
5. **Migration Safety** validated with rollback capability

### Quality Gates

- **Unit Test Coverage**: > 90%
- **Integration Test Pass Rate**: 100%
- **Performance Degradation**: < 100% increase
- **Migration Success Rate**: 100%
- **Docker Health Checks**: All passing

### Acceptance Validation

Each story must pass **all** "must" priority acceptance criteria and **80%** of "should" priority criteria to be considered complete.

## 📚 Additional Resources

- **Epic Documentation**: `OLLAMA_EPIC.md`
- **Implementation Stories**: `OLLAMA_STORY_*.md`
- **Architecture Guide**: `technical-specification.md`
- **Database Integration**: `database-agnostic-architecture.md`

## 🤝 Contributing

When adding new tests:

1. Follow the existing test structure and patterns
2. Add appropriate markers and fixtures
3. Include both positive and negative test cases
4. Update acceptance criteria mapping
5. Ensure tests are deterministic and fast

## 📞 Support

For issues with the testing framework:

1. Check environment variable configuration
2. Verify service availability (Ollama, Docker)
3. Review test logs in `test_reports/`
4. Run individual test suites to isolate issues
5. Consult the troubleshooting section above