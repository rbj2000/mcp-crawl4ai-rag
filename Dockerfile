FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files and README (required by pyproject.toml)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies based on provider
ARG VECTOR_DB_PROVIDER=supabase
RUN if [ "$VECTOR_DB_PROVIDER" = "all" ]; then \
        uv pip install --system -e ".[all]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "supabase" ]; then \
        uv pip install --system -e ".[supabase]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "pinecone" ]; then \
        uv pip install --system -e ".[pinecone]"; \
    elif [ "$VECTOR_DB_PROVIDER" = "weaviate" ]; then \
        uv pip install --system -e ".[weaviate]"; \
    else \
        uv pip install --system -e .; \
    fi

# Setup crawl4ai
RUN crawl4ai-setup

# Copy source code
COPY src/ ./src/
COPY knowledge_graphs/ ./knowledge_graphs/

# Set default environment variables
ENV HOST=0.0.0.0
ARG PORT=8051
ENV PORT=$PORT
ENV TRANSPORT=sse
ENV VECTOR_DB_PROVIDER=supabase

EXPOSE $PORT

CMD ["python", "src/crawl4ai_mcp_refactored.py"]
