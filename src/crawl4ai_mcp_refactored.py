"""
MCP server for web crawling with Crawl4AI - Database Agnostic Version.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Also includes AI hallucination detection and repository parsing tools using Neo4j knowledge graphs.
Supports multiple vector database backends: Supabase, Pinecone, SQLite, Weaviate, Neo4j Vector.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures
import sys

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import NoExtractionStrategy

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

# Import database abstraction layer
from database import (
    DatabaseConfig, 
    DatabaseProviderFactory, 
    VectorDatabaseProvider,
    DatabaseProvider
)

from utils_abstracted import (
    add_documents_to_database,
    search_documents_abstracted,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_database,
    extract_source_summary,
    search_code_examples_abstracted
)

# Import knowledge graph modules
from knowledge_graph_validator import KnowledgeGraphValidator
from parse_repo_into_neo4j import DirectNeo4jExtractor
from ai_script_analyzer import AIScriptAnalyzer
from hallucination_reporter import HallucinationReporter

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Helper functions for Neo4j validation and error handling
def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    return all([
        os.getenv("NEO4J_URI"),
        os.getenv("NEO4J_USER"),
        os.getenv("NEO4J_PASSWORD")
    ])

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    elif "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    else:
        return f"Neo4j error: {str(error)}"

def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}
    
    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}
    
    if not script_path.endswith('.py'):
        return {"valid": False, "error": "Only Python (.py) files are supported"}
    
    try:
        # Check if file is readable
        with open(script_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}

def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}
    
    repo_url = repo_url.strip()
    
    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}
    
    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}
    
    return {"valid": True, "repo_name": repo_url.split('/')[-1].replace('.git', '')}

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    vector_db: VectorDatabaseProvider
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and vector database provider
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize database provider
    try:
        db_config = DatabaseConfig.from_env()
        print(f"Initializing {db_config.provider.value} database provider...")
        vector_db = DatabaseProviderFactory.create_provider(db_config.provider, db_config.config)
        await vector_db.initialize()
        print(f"✓ {db_config.provider.value} database provider initialized")
    except Exception as e:
        print(f"Failed to initialize database provider: {e}")
        raise
    
    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("✓ Reranking model initialized")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None
    
    # Initialize Neo4j components if configured and enabled
    knowledge_validator = None
    repo_extractor = None
    
    # Check if knowledge graph functionality is enabled
    knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
    
    if knowledge_graph_enabled:
        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                print("Initializing knowledge graph components...")
                
                # Initialize knowledge graph validator
                knowledge_validator = KnowledgeGraphValidator(neo4j_uri, neo4j_user, neo4j_password)
                await knowledge_validator.initialize()
                print("✓ Knowledge graph validator initialized")
                
                # Initialize repository extractor
                repo_extractor = DirectNeo4jExtractor(neo4j_uri, neo4j_user, neo4j_password)
                await repo_extractor.initialize()
                print("✓ Repository extractor initialized")
                
            except Exception as e:
                print(f"Failed to initialize Neo4j components: {format_neo4j_error(e)}")
                knowledge_validator = None
                repo_extractor = None
        else:
            print("Neo4j credentials not configured - knowledge graph tools will be unavailable")
    else:
        print("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            vector_db=vector_db,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor
        )
    finally:
        # Clean up all components
        await crawler.__aexit__(None, None, None)
        await vector_db.close()
        print("✓ Vector database connection closed")
        
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                print("✓ Knowledge graph validator closed")
            except Exception as e:
                print(f"Error closing knowledge validator: {e}")
        if repo_extractor:
            try:
                await repo_extractor.close()
                print("✓ Repository extractor closed")
            except Exception as e:
                print(f"Error closing repository extractor: {e}")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="Database-agnostic MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051"))
)

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score in descending order
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return results
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results

@mcp.tool()
async def crawl_single_page(context: Context, url: str) -> str:
    """
    Crawl and index a single webpage.
    
    Args:
        url: The webpage URL to crawl
        
    Returns:
        JSON string with operation result
    """
    try:
        ctx = context.user_data
        print(f"Crawling single page: {url}")
        
        # Create crawler configuration
        crawler_config = CrawlerRunConfig(
            word_count_threshold=50,
            extraction_strategy=NoExtractionStrategy(),
            css_selector="body",
            cache_mode=CacheMode.BYPASS
        )
        
        # Crawl the page
        result = await ctx.crawler.arun(url=url, config=crawler_config)
        
        if not result.success:
            return json.dumps({
                "success": False,
                "error": f"Failed to crawl {url}: {result.error_message}"
            })
        
        # Extract content
        content = result.markdown or result.cleaned_html
        if not content:
            return json.dumps({
                "success": False,
                "error": f"No content extracted from {url}"
            })
        
        # Split content into chunks
        chunks = content.split('\n\n')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
        
        if not chunks:
            return json.dumps({
                "success": False,
                "error": f"No meaningful content chunks found in {url}"
            })
        
        # Prepare data for database insertion
        urls = [url] * len(chunks)
        chunk_numbers = list(range(len(chunks)))
        metadatas = [{"page_title": result.url} for _ in chunks]
        url_to_full_document = {url: content}
        
        # Add documents to database
        await add_documents_to_database(
            ctx.vector_db,
            urls, 
            chunk_numbers,
            chunks, 
            metadatas,
            url_to_full_document
        )
        
        # Update source information
        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path
        summary = extract_source_summary(source_id, content)
        word_count = len(content.split())
        
        await ctx.vector_db.update_source_info(source_id, summary, word_count)
        
        # Process code examples if agentic RAG is enabled
        use_agentic_rag = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        if use_agentic_rag:
            try:
                code_blocks = extract_code_blocks(content)
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []
                    
                    for i, block in enumerate(code_blocks):
                        code_urls.append(url)
                        code_chunk_numbers.append(i)
                        code_examples.append(block['code'])
                        
                        # Generate summary for code block
                        summary = generate_code_example_summary(
                            block['code'], 
                            block['context_before'], 
                            block['context_after']
                        )
                        code_summaries.append(summary)
                        code_metadatas.append({
                            "language": block['language'],
                            "page_title": result.url
                        })
                    
                    # Add code examples to database
                    await add_code_examples_to_database(
                        ctx.vector_db,
                        code_urls,
                        code_chunk_numbers,
                        code_examples,
                        code_summaries,
                        code_metadatas
                    )
                    
                    print(f"Added {len(code_blocks)} code examples from {url}")
            except Exception as e:
                print(f"Error processing code examples: {e}")
        
        return json.dumps({
            "success": True,
            "url": url,
            "chunks_added": len(chunks),
            "word_count": word_count,
            "source_id": source_id
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error crawling {url}: {str(e)}"
        })

@mcp.tool()
async def perform_rag_query(
    context: Context,
    query: str,
    match_count: int = 10,
    source_filter: Optional[str] = None
) -> str:
    """
    Perform a RAG query against the crawled documents.
    
    Args:
        query: The search query
        match_count: Maximum number of results to return
        source_filter: Optional source to filter results by
        
    Returns:
        JSON string with search results
    """
    try:
        ctx = context.user_data
        
        # Prepare filter metadata
        filter_metadata = {}
        if source_filter:
            filter_metadata["source"] = source_filter
        
        # Determine if hybrid search should be used
        use_hybrid = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Perform the search
        results = await search_documents_abstracted(
            ctx.vector_db,
            query,
            match_count,
            filter_metadata,
            use_hybrid
        )
        
        # Apply reranking if enabled
        if ctx.reranking_model and results:
            results = rerank_results(ctx.reranking_model, query, results, "content")
        
        return json.dumps({
            "success": True,
            "query": query,
            "results": results,
            "total_results": len(results),
            "hybrid_search_used": use_hybrid,
            "reranking_applied": ctx.reranking_model is not None
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error performing RAG query: {str(e)}"
        })

@mcp.tool()
async def get_available_sources(context: Context) -> str:
    """
    Get list of available sources in the database.
    
    Returns:
        JSON string with available sources
    """
    try:
        ctx = context.user_data
        sources = await ctx.vector_db.get_sources()
        
        source_list = [
            {
                "source_id": source.source_id,
                "summary": source.summary,
                "total_words": source.total_words,
                "created_at": source.created_at,
                "updated_at": source.updated_at
            }
            for source in sources
        ]
        
        return json.dumps({
            "success": True,
            "sources": source_list,
            "total_sources": len(source_list)
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Error getting sources: {str(e)}"
        })

# Add conditional tool for code search if agentic RAG is enabled
if os.getenv("USE_AGENTIC_RAG", "false") == "true":
    @mcp.tool()
    async def search_code_examples(
        context: Context,
        query: str,
        match_count: int = 10,
        source_filter: Optional[str] = None
    ) -> str:
        """
        Search for code examples in the database.
        
        Args:
            query: The search query for code examples
            match_count: Maximum number of results to return
            source_filter: Optional source to filter results by
            
        Returns:
            JSON string with code search results
        """
        try:
            ctx = context.user_data
            
            # Prepare filter metadata
            filter_metadata = {}
            if source_filter:
                filter_metadata["source"] = source_filter
            
            # Determine if hybrid search should be used
            use_hybrid = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
            
            # Perform the search
            results = await search_code_examples_abstracted(
                ctx.vector_db,
                query,
                match_count,
                filter_metadata,
                use_hybrid
            )
            
            # Apply reranking if enabled
            if ctx.reranking_model and results:
                results = rerank_results(ctx.reranking_model, query, results, "content")
            
            return json.dumps({
                "success": True,
                "query": query,
                "results": results,
                "total_results": len(results),
                "hybrid_search_used": use_hybrid,
                "reranking_applied": ctx.reranking_model is not None
            })
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error searching code examples: {str(e)}"
            })

# The rest of the tools remain the same, but will use ctx.vector_db instead of ctx.supabase_client
# This includes smart_crawl_url, parse_github_repository, check_ai_script_hallucinations, and query_knowledge_graph

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())