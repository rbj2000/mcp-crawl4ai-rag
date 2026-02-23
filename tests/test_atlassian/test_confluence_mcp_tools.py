"""Tests for Confluence MCP tools (Story 3.4).

Since MCP tool functions are defined inside a conditional ``if os.getenv("CONFLUENCE_URL")``
block, we set that env var and mock heavy dependencies before importing ``src.crawl4ai_mcp``,
then call the tool coroutines directly with a mocked MCP Context.
"""

import json
import os
import sys
import types
import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# 1. Set CONFLUENCE_URL so conditional tool block is registered on import.
# ---------------------------------------------------------------------------
os.environ["CONFLUENCE_URL"] = "https://test.atlassian.net"

# ---------------------------------------------------------------------------
# 2. Stub heavy third-party / project modules that crawl4ai_mcp imports at
#    module level so the test suite doesn't need Supabase, Crawl4AI, Neo4j…
# ---------------------------------------------------------------------------

def _stub_module(name, attrs=None):
    """Insert a fake module into sys.modules if it isn't already importable."""
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for attr, val in (attrs or {}).items():
        setattr(mod, attr, val)
    sys.modules[name] = mod


# sentence_transformers
_stub_module("sentence_transformers", {"CrossEncoder": MagicMock})

# crawl4ai tree
for _m in [
    "crawl4ai",
    "crawl4ai.async_configs",
]:
    _stub_module(_m)
# Provide names imported by crawl4ai_mcp
_crawl4ai = sys.modules["crawl4ai"]
for _name in [
    "AsyncWebCrawler",
    "BrowserConfig",
    "CrawlerRunConfig",
    "CacheMode",
    "MemoryAdaptiveDispatcher",
]:
    setattr(_crawl4ai, _name, MagicMock())

# supabase
_stub_module("supabase", {"Client": MagicMock, "create_client": MagicMock()})

# openai (used by src/utils.py)
_openai_mod = types.ModuleType("openai")
_openai_mod.api_key = None
sys.modules.setdefault("openai", _openai_mod)

# src.utils — stub so confluence_processor import works without real Supabase/OpenAI
_stub_module("src.utils", {
    "get_supabase_client": MagicMock(return_value=MagicMock()),
    "add_documents_to_supabase": MagicMock(),
    "search_documents": MagicMock(return_value=[]),
    "extract_code_blocks": MagicMock(return_value=[]),
    "generate_code_example_summary": MagicMock(return_value=""),
    "add_code_examples_to_supabase": MagicMock(),
    "update_source_info": MagicMock(),
    "extract_source_summary": MagicMock(return_value=""),
    "search_code_examples": MagicMock(return_value=[]),
})

# neo4j (used by knowledge graph modules)
_stub_module("neo4j", {"GraphDatabase": MagicMock})

# knowledge graph modules (imported via sys.path manipulation in crawl4ai_mcp)
for _kg_mod in [
    "knowledge_graph_validator",
    "parse_repo_into_neo4j",
    "ai_script_analyzer",
    "hallucination_reporter",
]:
    _stub_module(_kg_mod, {
        "KnowledgeGraphValidator": MagicMock,
        "DirectNeo4jExtractor": MagicMock,
        "AIScriptAnalyzer": MagicMock,
        "HallucinationReporter": MagicMock,
    })

# utils (imported as bare "utils" via the sys.path trick in crawl4ai_mcp)
_stub_module("utils", {
    "get_supabase_client": MagicMock(return_value=MagicMock()),
    "add_documents_to_supabase": MagicMock(),
    "search_documents": MagicMock(return_value=[]),
    "extract_code_blocks": MagicMock(return_value=[]),
    "generate_code_example_summary": MagicMock(return_value=""),
    "add_code_examples_to_supabase": MagicMock(),
    "update_source_info": MagicMock(),
    "extract_source_summary": MagicMock(return_value=""),
    "search_code_examples": MagicMock(return_value=[]),
})

# mcp.server.fastmcp — provide a stub FastMCP whose @tool() decorator is a no-op passthrough
class _FakeFastMCP:
    def __init__(self, *a, **kw):
        pass
    def tool(self, *a, **kw):
        """Decorator that simply returns the function unchanged."""
        def decorator(fn):
            return fn
        return decorator

_stub_module("mcp", {})
_stub_module("mcp.server", {})
_stub_module("mcp.server.fastmcp", {
    "FastMCP": _FakeFastMCP,
    "Context": type("Context", (), {}),
})

# dotenv
_stub_module("dotenv", {"load_dotenv": MagicMock()})

# Now we can import after all stubs are in place.
# We need to reload / freshly import crawl4ai_mcp since it may have been cached.
if "src.crawl4ai_mcp" in sys.modules:
    del sys.modules["src.crawl4ai_mcp"]

import src.crawl4ai_mcp as _mcp_mod  # noqa: E402

from src.atlassian.confluence_crawler import ConfluencePage, CrawlResult, CrawlSummary
from src.atlassian.confluence_processor import (
    ProcessingResult,
    ProcessingSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    page_id="101",
    title="Test Page",
    space_key="DEV",
    url="https://acme.atlassian.net/wiki/spaces/DEV/pages/101/Test-Page",
    content="# Hello\n\nSome content here.",
    last_modified="2025-06-01T00:00:00Z",
) -> ConfluencePage:
    return ConfluencePage(
        page_id=page_id,
        title=title,
        space_key=space_key,
        url=url,
        markdown_content=content,
        author="Alice",
        last_modified=last_modified,
        labels=["docs"],
        parent_page_id="100",
    )


def _make_crawl_result(pages=None, space_key="DEV", failed=0, errors=None):
    pages = pages or [_make_page()]
    summary = CrawlSummary(
        pages_succeeded=len(pages),
        pages_failed=failed,
        page_ids_crawled={p.page_id for p in pages},
        errors=errors or [],
    )
    return CrawlResult(pages=pages, summary=summary, space_key=space_key)


def _make_processing_result(
    pages_processed=1,
    chunks_stored=3,
    code_examples=0,
    pages_skipped_unchanged=0,
    pages_skipped_empty=0,
    pages_failed=0,
    orphaned_deleted=0,
    source_id="confluence:DEV",
):
    return ProcessingResult(
        page_results=[],
        summary=ProcessingSummary(
            pages_processed=pages_processed,
            pages_skipped_unchanged=pages_skipped_unchanged,
            pages_skipped_empty=pages_skipped_empty,
            pages_failed=pages_failed,
            total_chunks_stored=chunks_stored,
            total_code_examples_stored=code_examples,
            orphaned_pages_deleted=orphaned_deleted,
        ),
        source_id=source_id,
    )


@dataclass
class _FakeLifespan:
    confluence_crawler: Any = None
    confluence_processor: Any = None
    supabase_client: Any = None
    reranking_model: Any = None


class _FakeRequestContext:
    def __init__(self, lifespan):
        self.lifespan_context = lifespan


class _FakeContext:
    def __init__(self, lifespan):
        self.request_context = _FakeRequestContext(lifespan)


def _build_ctx(crawler=None, processor=None, supabase=None):
    lifespan = _FakeLifespan(
        confluence_crawler=crawler,
        confluence_processor=processor,
        supabase_client=supabase,
    )
    return _FakeContext(lifespan)


def _tool(name: str):
    """Get a tool function from the imported crawl4ai_mcp module."""
    return getattr(_mcp_mod, name)


# ===================================================================
# crawl_confluence_space
# ===================================================================


class TestCrawlConfluenceSpace:
    @pytest.mark.asyncio
    async def test_success(self):
        crawler = AsyncMock()
        processor = AsyncMock()
        crawl_result = _make_crawl_result()
        processing_result = _make_processing_result()

        crawler.crawl_space = AsyncMock(return_value=crawl_result)
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("crawl_confluence_space")
        raw = await fn(ctx, space_key="DEV", max_pages=500)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["space_key"] == "DEV"
        assert data["pages_crawled"] == 1
        assert data["chunks_stored"] == 3
        assert data["source_id"] == "confluence:DEV"
        crawler.crawl_space.assert_awaited_once_with("DEV", max_pages=500)

    @pytest.mark.asyncio
    async def test_partial_failure(self):
        pages = [_make_page(page_id="1"), _make_page(page_id="2")]
        errors = [{"page_id": "3", "error": "403 Forbidden"}]
        crawl_result = _make_crawl_result(pages=pages, failed=1, errors=errors)
        processing_result = _make_processing_result(pages_processed=2, chunks_stored=6)

        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_space = AsyncMock(return_value=crawl_result)
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("crawl_confluence_space")
        raw = await fn(ctx, space_key="DEV")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["pages_crawled"] == 2
        assert data["pages_failed"] == 1
        assert len(data["errors"]) == 1

    @pytest.mark.asyncio
    async def test_not_initialized(self):
        ctx = _build_ctx()
        fn = _tool("crawl_confluence_space")
        raw = await fn(ctx, space_key="DEV")
        data = json.loads(raw)

        assert data["success"] is False
        assert "not initialized" in data["error"]


# ===================================================================
# crawl_confluence_page
# ===================================================================


class TestCrawlConfluencePage:
    @pytest.mark.asyncio
    async def test_by_page_id(self):
        page = _make_page()
        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_page = AsyncMock(return_value=page)
        processing_result = _make_processing_result()
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("crawl_confluence_page")
        raw = await fn(ctx, page_id="101")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["page_id"] == "101"
        assert data["include_children"] is False
        crawler.crawl_page.assert_awaited_once_with("101")

    @pytest.mark.asyncio
    async def test_by_page_url(self):
        page = _make_page()
        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_page_by_url = AsyncMock(return_value=page)
        processing_result = _make_processing_result()
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("crawl_confluence_page")
        raw = await fn(ctx, page_url="https://acme.atlassian.net/wiki/spaces/DEV/pages/101/Test")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["page_id"] == "101"

    @pytest.mark.asyncio
    async def test_with_children(self):
        crawl_result = _make_crawl_result(
            pages=[_make_page(page_id="101"), _make_page(page_id="102")]
        )
        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_page_tree = AsyncMock(return_value=crawl_result)
        processing_result = _make_processing_result(pages_processed=2, chunks_stored=6)
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("crawl_confluence_page")
        raw = await fn(ctx, page_id="101", include_children=True, max_depth=3)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["include_children"] is True
        assert data["pages_crawled"] == 2
        crawler.crawl_page_tree.assert_awaited_once_with("101", max_depth=3)

    @pytest.mark.asyncio
    async def test_neither_id_nor_url(self):
        ctx = _build_ctx(crawler=AsyncMock(), processor=AsyncMock())
        fn = _tool("crawl_confluence_page")
        raw = await fn(ctx)
        data = json.loads(raw)

        assert data["success"] is False
        assert "page_id or page_url" in data["error"]


# ===================================================================
# get_confluence_sources
# ===================================================================


class TestGetConfluenceSources:
    @pytest.mark.asyncio
    async def test_returns_sources(self):
        mock_client = MagicMock()

        source_data = [
            {
                "source_id": "confluence:DEV",
                "summary": "Dev docs",
                "total_words": 5000,
                "created_at": "2025-01-01",
                "updated_at": "2025-06-01",
            }
        ]
        source_exec = MagicMock()
        source_exec.data = source_data

        sources_chain = MagicMock()
        sources_chain.select.return_value = sources_chain
        sources_chain.like.return_value = sources_chain
        sources_chain.order.return_value = sources_chain
        sources_chain.execute.return_value = source_exec

        chunk_exec = MagicMock()
        chunk_exec.count = 42
        chunk_chain = MagicMock()
        chunk_chain.select.return_value = chunk_chain
        chunk_chain.eq.return_value = chunk_chain
        chunk_chain.execute.return_value = chunk_exec

        mock_client.from_.side_effect = [sources_chain, chunk_chain]

        ctx = _build_ctx(supabase=mock_client)
        fn = _tool("get_confluence_sources")
        raw = await fn(ctx)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["count"] == 1
        assert data["sources"][0]["source_id"] == "confluence:DEV"
        assert data["sources"][0]["chunk_count"] == 42

    @pytest.mark.asyncio
    async def test_no_sources(self):
        mock_client = MagicMock()
        exec_result = MagicMock()
        exec_result.data = []

        chain = MagicMock()
        chain.select.return_value = chain
        chain.like.return_value = chain
        chain.order.return_value = chain
        chain.execute.return_value = exec_result
        mock_client.from_.return_value = chain

        ctx = _build_ctx(supabase=mock_client)
        fn = _tool("get_confluence_sources")
        raw = await fn(ctx)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["count"] == 0
        assert data["sources"] == []


# ===================================================================
# sync_confluence_space
# ===================================================================


class TestSyncConfluenceSpace:
    @pytest.mark.asyncio
    async def test_incremental(self):
        crawl_result = _make_crawl_result()
        processing_result = _make_processing_result(
            pages_processed=0,
            pages_skipped_unchanged=1,
            chunks_stored=0,
        )

        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_space = AsyncMock(return_value=crawl_result)
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("sync_confluence_space")
        raw = await fn(ctx, space_key="DEV")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["force_full"] is False
        assert data["pages_unchanged"] == 1
        assert data["pages_updated"] == 0
        processor.process_crawl_result.assert_awaited_once_with(
            crawl_result, detect_deletions=False
        )

    @pytest.mark.asyncio
    async def test_force_full_with_deletions(self):
        crawl_result = _make_crawl_result()
        processing_result = _make_processing_result(
            pages_processed=1,
            chunks_stored=3,
            orphaned_deleted=2,
        )

        crawler = AsyncMock()
        processor = AsyncMock()
        crawler.crawl_space = AsyncMock(return_value=crawl_result)
        processor.process_crawl_result = AsyncMock(return_value=processing_result)

        ctx = _build_ctx(crawler=crawler, processor=processor)
        fn = _tool("sync_confluence_space")
        raw = await fn(ctx, space_key="DEV", force_full=True)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["force_full"] is True
        assert data["orphaned_deleted"] == 2
        processor.process_crawl_result.assert_awaited_once_with(
            crawl_result, detect_deletions=True
        )


# ===================================================================
# perform_rag_query — source_type filter
# ===================================================================


class TestPerformRagQuerySourceType:
    @pytest.mark.asyncio
    async def test_confluence_filter(self):
        """source_type='confluence' should only return confluence: prefixed results."""
        mock_search = MagicMock(return_value=[
            {"id": "1", "url": "https://a.com", "content": "web", "metadata": {}, "similarity": 0.9, "source_id": "a.com"},
            {"id": "2", "url": "https://b.com", "content": "conf", "metadata": {}, "similarity": 0.8, "source_id": "confluence:DEV"},
        ])

        lifespan = _FakeLifespan(supabase_client=MagicMock())
        lifespan.reranking_model = None
        ctx = _FakeContext(lifespan)

        fn = _tool("perform_rag_query")
        with patch.object(_mcp_mod, "search_documents", mock_search):
            raw = await fn(ctx, query="test", source_type="confluence")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["source_type_filter"] == "confluence"
        assert len(data["results"]) == 1
        assert data["results"][0]["url"] == "https://b.com"

    @pytest.mark.asyncio
    async def test_no_source_type_returns_all(self):
        """source_type=None should return all results (backward compatible)."""
        mock_search = MagicMock(return_value=[
            {"id": "1", "url": "https://a.com", "content": "web", "metadata": {}, "similarity": 0.9, "source_id": "a.com"},
            {"id": "2", "url": "https://b.com", "content": "conf", "metadata": {}, "similarity": 0.8, "source_id": "confluence:DEV"},
        ])

        lifespan = _FakeLifespan(supabase_client=MagicMock())
        lifespan.reranking_model = None
        ctx = _FakeContext(lifespan)

        fn = _tool("perform_rag_query")
        with patch.object(_mcp_mod, "search_documents", mock_search):
            raw = await fn(ctx, query="test")
        data = json.loads(raw)

        assert data["success"] is True
        assert data["source_type_filter"] is None
        assert len(data["results"]) == 2
