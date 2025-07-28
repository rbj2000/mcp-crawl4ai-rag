FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies based on providers
ARG VECTOR_DB_PROVIDER=supabase
ARG AI_PROVIDER=openai

RUN if [ "$VECTOR_DB_PROVIDER" = "all" ]; then \
        uv pip install --system -e ".[all]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "supabase" ]; then \
        uv pip install --system -e ".[supabase]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "pinecone" ]; then \
        uv pip install --system -e ".[pinecone]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "weaviate" ]; then \
        uv pip install --system -e ".[weaviate]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "neo4j_vector" ]; then \
        uv pip install --system -e ".[neo4j]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "sqlite" ]; then \
        uv pip install --system -e ".[sqlite]"; \
    else \
        uv pip install --system -e .; \
    fi

# Install AI provider specific dependencies
RUN if [ "$AI_PROVIDER" = "ollama" ] || [ "$AI_PROVIDER" = "mixed" ]; then \
        uv pip install --system ollama requests; \
    fi

# Install monitoring dependencies
RUN uv pip install --system prometheus-client psutil

# Setup crawl4ai
RUN crawl4ai-setup

# Copy source code
COPY src/ ./src/
COPY knowledge_graphs/ ./knowledge_graphs/

# Create health check script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Health check function\n\
health_check() {\n\
    local endpoint="${1:-http://localhost:8051/health}"\n\
    local timeout="${2:-10}"\n\
    \n\
    if command -v curl >/dev/null 2>&1; then\n\
        curl -f -s --max-time "${timeout}" "${endpoint}" >/dev/null\n\
    else\n\
        echo "curl not available for health check"\n\
        exit 1\n\
    fi\n\
}\n\
\n\
# Provider connectivity checks\n\
check_providers() {\n\
    if [ "${AI_PROVIDER}" = "ollama" ] || [ "${AI_PROVIDER}" = "mixed" ]; then\n\
        local ollama_host=$(echo "${OLLAMA_BASE_URL:-http://localhost:11434}" | sed "s|http://||" | cut -d: -f1)\n\
        local ollama_port=$(echo "${OLLAMA_BASE_URL:-http://localhost:11434}" | sed "s|http://||" | cut -d: -f2)\n\
        ollama_port=${ollama_port:-11434}\n\
        \n\
        if ! nc -z "${ollama_host}" "${ollama_port}"; then\n\
            echo "Warning: Ollama service not reachable at ${ollama_host}:${ollama_port}"\n\
        fi\n\
    fi\n\
}\n\
\n\
# Run checks\n\
check_providers\n\
health_check "$@"' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Set default environment variables
ENV HOST=0.0.0.0
ARG PORT=8051
ENV PORT=$PORT
ENV TRANSPORT=sse
ENV VECTOR_DB_PROVIDER=supabase
ENV AI_PROVIDER=openai

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

EXPOSE $PORT

CMD ["python", "src/crawl4ai_mcp_refactored.py"]
