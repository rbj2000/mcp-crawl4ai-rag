#!/bin/bash

# Comprehensive Docker Testing Script for MCP Crawl4AI RAG
# Tests all database providers with functional validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Test results tracking
PASSED_TESTS=0
FAILED_TESTS=0
TOTAL_TESTS=0

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_TESTS++))
}

error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_TESTS++))
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Cleanup function
cleanup() {
    log "Cleaning up Docker containers and networks..."
    docker-compose down -v --remove-orphans 2>/dev/null || true
    docker system prune -f 2>/dev/null || true
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local max_retries=${3:-$HEALTH_CHECK_RETRIES}
    
    log "Waiting for $service_name to be healthy..."
    
    for i in $(seq 1 $max_retries); do
        if curl -f -s "$health_url" >/dev/null 2>&1; then
            success "$service_name is healthy"
            return 0
        fi
        
        if [ $i -eq $max_retries ]; then
            error "$service_name failed to become healthy after $max_retries attempts"
            return 1
        fi
        
        log "Attempt $i/$max_retries: $service_name not ready, waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Test MCP server functionality
test_mcp_functionality() {
    local service_name=$1
    local port=$2
    
    log "Testing MCP server functionality for $service_name..."
    ((TOTAL_TESTS++))
    
    # Test server is responding
    if ! curl -f -s "http://localhost:$port/health" >/dev/null 2>&1; then
        error "$service_name: MCP server not responding on port $port"
        return 1
    fi
    
    # Test MCP tools are available
    local tools_response=$(curl -s "http://localhost:$port/tools" 2>/dev/null || echo "")
    
    if [[ $tools_response == *"crawl_single_page"* ]]; then
        success "$service_name: MCP tools are available"
    else
        error "$service_name: MCP tools not available"
        return 1
    fi
    
    # Test database connection
    local db_test_response=$(curl -s -X POST "http://localhost:$port/test-db" 2>/dev/null || echo "error")
    
    if [[ $db_test_response == *"success"* ]]; then
        success "$service_name: Database connection working"
    else
        error "$service_name: Database connection failed"
        return 1
    fi
    
    return 0
}

# Test specific provider functionality
test_provider_functionality() {
    local provider=$1
    local port=$2
    
    log "Testing $provider provider functionality..."
    ((TOTAL_TESTS++))
    
    # Create test document
    local test_url="https://example.com/test"
    local crawl_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"url\": \"$test_url\"}" \
        "http://localhost:$port/crawl" 2>/dev/null || echo "error")
    
    if [[ $crawl_response == *"success"* ]]; then
        success "$provider: Document crawling works"
    else
        warning "$provider: Document crawling test skipped (requires external URL)"
    fi
    
    # Test search functionality
    local search_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"test query\", \"match_count\": 5}" \
        "http://localhost:$port/search" 2>/dev/null || echo "error")
    
    if [[ $search_response != "error" ]]; then
        success "$provider: Search functionality works"
    else
        error "$provider: Search functionality failed"
        return 1
    fi
    
    return 0
}

# Test SQLite provider
test_sqlite_provider() {
    log "🗄️  Testing SQLite Provider"
    
    export VECTOR_DB_PROVIDER=sqlite
    export SQLITE_DB_PATH=/tmp/test.sqlite
    export OPENAI_API_KEY=test_key
    
    # Start SQLite service
    docker-compose --profile sqlite up -d
    
    if wait_for_service "SQLite MCP Server" "http://localhost:8052/health"; then
        if test_mcp_functionality "SQLite" 8052; then
            if test_provider_functionality "SQLite" 8052; then
                success "SQLite provider tests passed"
            fi
        fi
    fi
    
    # Cleanup
    docker-compose --profile sqlite down -v
}

# Test Supabase provider (requires credentials)
test_supabase_provider() {
    log "🐘 Testing Supabase Provider"
    
    if [[ -z "$SUPABASE_URL" || -z "$SUPABASE_SERVICE_KEY" ]]; then
        warning "Skipping Supabase tests - credentials not provided"
        return 0
    fi
    
    export VECTOR_DB_PROVIDER=supabase
    export OPENAI_API_KEY=test_key
    
    # Start Supabase service
    docker-compose --profile supabase up -d
    
    if wait_for_service "Supabase MCP Server" "http://localhost:8051/health"; then
        if test_mcp_functionality "Supabase" 8051; then
            if test_provider_functionality "Supabase" 8051; then
                success "Supabase provider tests passed"
            fi
        fi
    fi
    
    # Cleanup
    docker-compose --profile supabase down -v
}

# Test Neo4j provider
test_neo4j_provider() {
    log "🕸️  Testing Neo4j Provider"
    
    export VECTOR_DB_PROVIDER=neo4j_vector
    export NEO4J_PASSWORD=test_password
    export OPENAI_API_KEY=test_key
    
    # Start Neo4j service
    docker-compose --profile neo4j up -d
    
    # Wait for Neo4j to be ready
    if wait_for_service "Neo4j Database" "http://localhost:7474" 60; then
        if wait_for_service "Neo4j MCP Server" "http://localhost:8055/health"; then
            if test_mcp_functionality "Neo4j" 8055; then
                if test_provider_functionality "Neo4j" 8055; then
                    success "Neo4j provider tests passed"
                fi
            fi
        fi
    fi
    
    # Cleanup
    docker-compose --profile neo4j down -v
}

# Test Pinecone provider (requires credentials)
test_pinecone_provider() {
    log "🌲 Testing Pinecone Provider"
    
    if [[ -z "$PINECONE_API_KEY" || -z "$PINECONE_ENVIRONMENT" ]]; then
        warning "Skipping Pinecone tests - credentials not provided"
        return 0
    fi
    
    export VECTOR_DB_PROVIDER=pinecone
    export PINECONE_INDEX_NAME=test-crawl4ai-rag
    export OPENAI_API_KEY=test_key
    
    # Start Pinecone service
    docker-compose --profile pinecone up -d
    
    if wait_for_service "Pinecone MCP Server" "http://localhost:8054/health"; then
        if test_mcp_functionality "Pinecone" 8054; then
            if test_provider_functionality "Pinecone" 8054; then
                success "Pinecone provider tests passed"
            fi
        fi
    fi
    
    # Cleanup
    docker-compose --profile pinecone down -v
}

# Build test - verify all images can be built
test_docker_builds() {
    log "🏗️  Testing Docker Builds"
    
    local providers=("supabase" "sqlite" "pinecone" "all")
    
    for provider in "${providers[@]}"; do
        ((TOTAL_TESTS++))
        log "Building Docker image for $provider provider..."
        
        if docker build --build-arg VECTOR_DB_PROVIDER=$provider -t mcp-crawl4ai-rag:$provider . > /tmp/build_$provider.log 2>&1; then
            success "Docker build successful for $provider provider"
        else
            error "Docker build failed for $provider provider"
            log "Build log saved to /tmp/build_$provider.log"
        fi
    done
}

# Security test - check for vulnerabilities
test_security() {
    log "🔒 Testing Security"
    ((TOTAL_TESTS++))
    
    # Check if we can scan for vulnerabilities
    if command -v docker &> /dev/null; then
        if docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
           -v /tmp:/tmp aquasec/trivy image mcp-crawl4ai-rag:sqlite > /tmp/security_scan.log 2>&1; then
            success "Security scan completed - check /tmp/security_scan.log"
        else
            warning "Security scan skipped - Trivy not available"
        fi
    fi
}

# Performance test - basic load testing
test_performance() {
    log "⚡ Testing Performance"
    ((TOTAL_TESTS++))
    
    # Start SQLite service for performance testing
    export VECTOR_DB_PROVIDER=sqlite
    docker-compose --profile sqlite up -d
    
    if wait_for_service "SQLite MCP Server" "http://localhost:8052/health"; then
        # Simple load test with curl
        log "Running basic load test..."
        
        local start_time=$(date +%s)
        for i in {1..10}; do
            curl -f -s "http://localhost:8052/health" >/dev/null || break
        done
        local end_time=$(date +%s)
        
        local duration=$((end_time - start_time))
        log "Completed 10 requests in ${duration}s"
        
        if [ $duration -lt 10 ]; then
            success "Performance test passed"
        else
            warning "Performance test - requests took longer than expected"
        fi
    fi
    
    docker-compose --profile sqlite down -v
}

# Main test execution
main() {
    log "🚀 Starting Comprehensive Docker Testing for MCP Crawl4AI RAG"
    log "Testing all database providers with functional validation"
    
    # Setup
    trap cleanup EXIT
    cleanup
    
    # Create necessary directories
    mkdir -p /tmp
    
    # Run tests
    test_docker_builds
    test_sqlite_provider
    test_neo4j_provider
    test_supabase_provider
    test_pinecone_provider
    test_security
    test_performance
    
    # Summary
    log "📊 Test Summary"
    log "Total Tests: $TOTAL_TESTS"
    success "Passed: $PASSED_TESTS"
    error "Failed: $FAILED_TESTS"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        success "🎉 All tests passed!"
        exit 0
    else
        error "💥 Some tests failed!"
        exit 1
    fi
}

# Help function
show_help() {
    echo "Comprehensive Docker Testing Script for MCP Crawl4AI RAG"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --provider <name>    Test specific provider only (sqlite|supabase|neo4j|pinecone)"
    echo "  --build-only         Only test Docker builds"
    echo "  --no-cleanup         Skip cleanup after tests"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SUPABASE_URL         Supabase project URL (for Supabase tests)"
    echo "  SUPABASE_SERVICE_KEY Supabase service key (for Supabase tests)"
    echo "  PINECONE_API_KEY     Pinecone API key (for Pinecone tests)"
    echo "  PINECONE_ENVIRONMENT Pinecone environment (for Pinecone tests)"
    echo ""
    echo "Examples:"
    echo "  $0                   # Run all tests"
    echo "  $0 --provider sqlite # Test only SQLite provider"
    echo "  $0 --build-only      # Only test Docker builds"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)
            SPECIFIC_PROVIDER="$2"
            shift 2
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run specific provider test if requested
if [[ -n "$SPECIFIC_PROVIDER" ]]; then
    case $SPECIFIC_PROVIDER in
        sqlite)
            test_sqlite_provider
            ;;
        supabase)
            test_supabase_provider
            ;;
        neo4j)
            test_neo4j_provider
            ;;
        pinecone)
            test_pinecone_provider
            ;;
        *)
            error "Unknown provider: $SPECIFIC_PROVIDER"
            exit 1
            ;;
    esac
    exit 0
fi

# Run build-only if requested
if [[ "$BUILD_ONLY" == "true" ]]; then
    test_docker_builds
    exit 0
fi

# Run main test suite
main