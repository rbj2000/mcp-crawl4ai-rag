# Neo4j Integration Guide

This guide explains how to use Neo4j as both a vector database and knowledge graph in the mcp-crawl4ai-rag system.

## Overview

Neo4j can serve dual purposes in this system:

1. **Vector Database** (`VECTOR_DB_PROVIDER=neo4j_vector`): Store and search document embeddings using Neo4j's vector indexes
2. **Knowledge Graph** (`USE_KNOWLEDGE_GRAPH=true`): Store and query repository structure for AI hallucination detection

## Architecture Benefits

Using Neo4j for both purposes provides several advantages:

- **Unified Storage**: Documents, code examples, and knowledge graph data in one database
- **Rich Relationships**: Connect documents to code structures through graph relationships
- **Advanced Queries**: Combine vector search with graph traversals
- **Single Infrastructure**: Reduce complexity by using one database technology

## Setup Instructions

### 1. Start Neo4j Database

#### Using Docker Compose (Recommended)
```bash
# Start Neo4j with the MCP server
docker-compose --profile neo4j up

# This starts:
# - Neo4j database on ports 7474 (web) and 7687 (bolt)
# - MCP server on port 8055
```

#### Using Local Neo4j Installation
```bash
# Install Neo4j locally
# Download from https://neo4j.com/download/

# Start Neo4j
neo4j start
```

### 2. Configure Environment Variables

```bash
# Set Neo4j as the vector database provider
VECTOR_DB_PROVIDER=neo4j_vector

# Vector database configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Knowledge graph configuration (can use same instance)
KG_NEO4J_URI=bolt://localhost:7687
KG_NEO4J_USER=neo4j
KG_NEO4J_PASSWORD=your_password
KG_NEO4J_DATABASE=neo4j

# Enable knowledge graph features
USE_KNOWLEDGE_GRAPH=true
USE_HYBRID_SEARCH=true  # Neo4j supports excellent hybrid search
```

### 3. Install Dependencies

```bash
# Install with Neo4j support (it's already included in base dependencies)
uv pip install -e .

# Or if you want all database providers
uv pip install -e ".[all]"
```

## Database Schema

### Vector Database Schema

Neo4j creates the following node types and indexes for vector operations:

```cypher
# Node Labels
(:Document)     # Document chunks with embeddings
(:CodeExample)  # Code examples with embeddings  
(:Source)       # Source metadata

# Vector Indexes
document_embeddings  # Vector index on Document.embedding
code_embeddings     # Vector index on CodeExample.embedding

# Regular Indexes
document_source_id  # Index on Document.source_id
document_url       # Index on Document.url
code_source_id     # Index on CodeExample.source_id
code_url          # Index on CodeExample.url
```

### Knowledge Graph Schema

The knowledge graph uses separate node labels that don't conflict with vector storage:

```cypher
# Knowledge Graph Nodes
(:Repository)   # GitHub repositories
(:File)        # Python files
(:Class)       # Python classes
(:Function)    # Python functions
(:Method)      # Class methods
(:Import)      # Import statements

# Knowledge Graph Relationships
(Repository)-[:CONTAINS]->(File)
(File)-[:DEFINES]->(Class)
(File)-[:DEFINES]->(Function)
(Class)-[:HAS_METHOD]->(Method)
(File)-[:IMPORTS]->(Import)
```

## Usage Examples

### 1. Basic Vector Search

```python
# The MCP server automatically uses Neo4j for:
# - Document storage and retrieval
# - Code example search
# - Hybrid search (vector + keyword)

# Example MCP tool call:
{
    "method": "perform_rag_query",
    "params": {
        "query": "How to implement async functions",
        "match_count": 10,
        "source_filter": "python.org"
    }
}
```

### 2. Combined Vector + Knowledge Graph Queries

```cypher
# Find documents related to specific functions in the knowledge graph
MATCH (f:Function {name: "async def process_data"})
MATCH (d:Document)
WHERE d.source_id CONTAINS f.file_name
CALL db.index.vector.queryNodes('document_embeddings', 5, $query_embedding)
YIELD node, score
WHERE node = d
RETURN d.content, score
```

### 3. Hybrid Search with Graph Context

```cypher
# Search for code examples and enrich with repository context
CALL db.index.vector.queryNodes('code_embeddings', 10, $query_embedding)
YIELD node AS c, score
OPTIONAL MATCH (r:Repository)-[:CONTAINS]->(f:File)
WHERE f.path CONTAINS c.url
RETURN c.content, c.summary, r.name as repo_name, score
ORDER BY score DESC
```

## Advanced Features

### 1. Cross-Modal Search

Combine document and code searches in a single query:

```cypher
# Find both documents and code examples for a topic
(CALL db.index.vector.queryNodes('document_embeddings', 5, $query_embedding)
 YIELD node AS doc, score AS doc_score
 RETURN doc.content as content, 'document' as type, doc_score as score)
UNION
(CALL db.index.vector.queryNodes('code_embeddings', 5, $query_embedding)
 YIELD node AS code, score AS code_score
 RETURN code.content as content, 'code' as type, code_score as score)
ORDER BY score DESC
LIMIT 10
```

### 2. Hallucination Detection with Context

```cypher
# Verify if mentioned functions actually exist in the codebase
MATCH (f:Function)
WHERE f.name CONTAINS $mentioned_function
OPTIONAL MATCH (f)<-[:HAS_METHOD]-(c:Class)
OPTIONAL MATCH (f)<-[:DEFINES]-(file:File)
RETURN f.name, c.name as class_name, file.path,
       f.parameters, f.docstring
```

### 3. Source Relationship Analysis

```cypher
# Find related documents through knowledge graph connections
MATCH (d:Document)
WHERE d.url = $document_url
MATCH (f:File)
WHERE f.path CONTAINS d.source_id
MATCH (f)-[:DEFINES]->(func:Function)
MATCH (other_file:File)-[:DEFINES]->(related_func:Function)
WHERE related_func.name CONTAINS func.name
MATCH (other_doc:Document)
WHERE other_doc.source_id CONTAINS other_file.path
RETURN other_doc.content, other_doc.url
```

## Performance Considerations

### Vector Index Optimization

```cypher
# Monitor vector index performance
SHOW INDEXES
WHERE type = "VECTOR"

# Check index statistics
CALL db.index.vector.queryNodes('document_embeddings', 1, [0.1] * 1536)
YIELD node, score
RETURN count(*) as total_vectors
```

### Query Performance

- **Batch Operations**: Neo4j handles batch inserts efficiently with UNWIND
- **Index Usage**: Both vector and traditional indexes are used automatically
- **Memory Management**: Configure Neo4j heap size based on your dataset
- **Connection Pooling**: The driver handles connection pooling automatically

## Monitoring and Debugging

### 1. Neo4j Browser

Access the Neo4j Browser at `http://localhost:7474` to:
- Visualize the graph structure
- Run custom Cypher queries
- Monitor performance metrics
- Explore relationships between documents and code

### 2. Query Debugging

```cypher
# Check document distribution by source
MATCH (d:Document)
RETURN d.source_id, count(d) as doc_count
ORDER BY doc_count DESC

# Verify vector embeddings
MATCH (d:Document)
WHERE size(d.embedding) > 0
RETURN count(d) as documents_with_embeddings

# Check knowledge graph completeness
MATCH (r:Repository)-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
RETURN r.name, count(func) as function_count
ORDER BY function_count DESC
```

### 3. Performance Monitoring

```cypher
# Monitor query performance
PROFILE CALL db.index.vector.queryNodes('document_embeddings', 10, $embedding)
YIELD node, score
RETURN node.content, score

# Check memory usage
CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Memory Pools")
YIELD attributes
RETURN attributes.Name, attributes.Usage
```

## Migration from Other Providers

### From Supabase to Neo4j

1. Export your existing data from Supabase
2. Set `VECTOR_DB_PROVIDER=neo4j_vector`
3. Re-crawl your sources (the system will automatically store in Neo4j)
4. Optionally: Enable knowledge graph features for enhanced capabilities

### Data Migration Script

```python
# Example migration helper
async def migrate_to_neo4j():
    # Initialize both providers
    old_provider = SupabaseProvider(supabase_config)
    new_provider = Neo4jVectorProvider(neo4j_config)
    
    # Get all sources from old provider
    sources = await old_provider.get_sources()
    
    # For each source, search all documents and migrate
    for source in sources:
        docs = await old_provider.search_documents(
            query_embedding=[0] * 1536,  # Dummy embedding
            match_count=10000,
            filter_metadata={"source": source.source_id}
        )
        # Convert and migrate documents...
```

## Troubleshooting

### Common Issues

1. **Vector index not found**: Ensure Neo4j 5.x+ and indexes are created
2. **Connection refused**: Check Neo4j is running and bolt port is accessible
3. **Authentication failed**: Verify username/password configuration
4. **Slow queries**: Check vector index creation and heap size configuration

### Debug Steps

```bash
# Check Neo4j status
neo4j status

# View logs
tail -f /path/to/neo4j/logs/neo4j.log

# Test connection
cypher-shell -u neo4j -p your_password "RETURN 1"
```

Neo4j provides a powerful, unified solution for both vector storage and knowledge graph capabilities, enabling sophisticated RAG applications with rich contextual understanding! 🚀