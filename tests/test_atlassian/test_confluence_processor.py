"""Tests for ConfluenceProcessor — mocked at the utils function level."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.atlassian.confluence_crawler import ConfluencePage, CrawlResult, CrawlSummary
from src.atlassian.confluence_processor import (
    ConfluenceProcessor,
    PageProcessingStatus,
    PageResult,
    ProcessingResult,
    ProcessingSummary,
    _smart_chunk_markdown,
    _extract_section_info,
)


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

def _make_page(
    page_id="101",
    title="Test Page",
    space_key="DEV",
    url="https://acme.atlassian.net/wiki/spaces/DEV/pages/101/Test-Page",
    content="# Hello\n\nSome content here.",
    author="Alice",
    last_modified="2025-06-01T00:00:00Z",
    labels=None,
    parent_page_id="100",
) -> ConfluencePage:
    return ConfluencePage(
        page_id=page_id,
        title=title,
        space_key=space_key,
        url=url,
        markdown_content=content,
        author=author,
        last_modified=last_modified,
        labels=labels or ["docs"],
        parent_page_id=parent_page_id,
    )


def _make_crawl_result(pages=None, space_key="DEV") -> CrawlResult:
    return CrawlResult(
        pages=pages or [_make_page()],
        summary=CrawlSummary(pages_succeeded=len(pages or [_make_page()])),
        space_key=space_key,
    )


def _mock_supabase_client():
    """Return a mock Supabase client with chainable table methods."""
    client = MagicMock()
    # Default: queries return empty data (no existing rows)
    select_chain = MagicMock()
    select_chain.execute.return_value = MagicMock(data=[])
    table_chain = MagicMock()
    table_chain.select.return_value = select_chain
    select_chain.eq.return_value = select_chain
    select_chain.limit.return_value = select_chain
    client.table.return_value = table_chain
    return client


# Shared patch targets — patch on src.utils (referenced via _utils in processor)
_PATCH_ADD_DOCS = "src.utils.add_documents_to_supabase"
_PATCH_ADD_CODE = "src.utils.add_code_examples_to_supabase"
_PATCH_UPDATE_SOURCE = "src.utils.update_source_info"
_PATCH_EXTRACT_SUMMARY = "src.utils.extract_source_summary"
_PATCH_EXTRACT_CODE = "src.utils.extract_code_blocks"
_PATCH_GEN_CODE_SUMMARY = "src.utils.generate_code_example_summary"


# Convenience decorator that patches all utils used by the processor
def _patch_all_utils():
    """Returns a list of patch objects for all utils; use as context managers or decorators."""
    return [
        patch(_PATCH_ADD_DOCS),
        patch(_PATCH_UPDATE_SOURCE),
        patch(_PATCH_EXTRACT_SUMMARY, return_value="A great Confluence space."),
        patch(_PATCH_EXTRACT_CODE, return_value=[]),
        patch(_PATCH_GEN_CODE_SUMMARY, return_value="Code example summary."),
        patch(_PATCH_ADD_CODE),
    ]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDataclassDefaults:
    """Dataclass default values are sensible."""

    def test_page_result_defaults(self):
        pr = PageResult(page_id="1", title="T", url="u", status=PageProcessingStatus.STORED)
        assert pr.chunks_stored == 0
        assert pr.code_examples_stored == 0
        assert pr.error is None

    def test_processing_summary_defaults(self):
        s = ProcessingSummary()
        assert s.pages_processed == 0
        assert s.pages_skipped_unchanged == 0
        assert s.pages_skipped_empty == 0
        assert s.pages_failed == 0
        assert s.total_chunks_stored == 0
        assert s.total_code_examples_stored == 0
        assert s.orphaned_pages_deleted == 0

    def test_processing_result_defaults(self):
        r = ProcessingResult()
        assert r.page_results == []
        assert r.source_id is None


class TestSourceIdFormat:
    """Source ID follows the `confluence:{space_key}` convention."""

    @pytest.mark.asyncio
    async def test_source_id_format(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        crawl = _make_crawl_result(space_key="MYSPACE")

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_UPDATE_SOURCE) as mock_update, \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"):
            result = await processor.process_crawl_result(crawl)

        assert result.source_id == "confluence:MYSPACE"

    @pytest.mark.asyncio
    async def test_source_id_inferred_from_page(self):
        """When CrawlResult.space_key is None, infer from first page."""
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        crawl = _make_crawl_result(space_key=None)
        # space_key is None on the CrawlResult but pages have it

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_UPDATE_SOURCE), \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"):
            result = await processor.process_crawl_result(crawl)

        assert result.source_id == "confluence:DEV"


class TestSinglePageProcessing:
    """Single page is chunked, embedded, and stored correctly."""

    @pytest.mark.asyncio
    async def test_stores_chunks_with_confluence_metadata(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="# Title\n\nParagraph one.\n\nParagraph two.")

        with patch(_PATCH_ADD_DOCS) as mock_add:
            page_result = await processor.process_page(page, "confluence:DEV")

        assert page_result.status == PageProcessingStatus.STORED
        assert page_result.chunks_stored >= 1

        # Verify add_documents_to_supabase was called
        mock_add.assert_called_once()
        call_args = mock_add.call_args

        # Check metadata includes Confluence-specific fields
        metadatas = call_args[0][4]  # 5th positional arg
        meta = metadatas[0]
        assert meta["source_type"] == "confluence"
        assert meta["space_key"] == "DEV"
        assert meta["page_id"] == "101"
        assert meta["page_title"] == "Test Page"
        assert meta["labels"] == ["docs"]
        assert meta["author"] == "Alice"
        assert meta["last_modified"] == "2025-06-01T00:00:00Z"
        assert meta["parent_page_id"] == "100"
        assert meta["source"] == "confluence:DEV"

    @pytest.mark.asyncio
    async def test_url_to_full_document_passed(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="Full document content.")

        with patch(_PATCH_ADD_DOCS) as mock_add:
            await processor.process_page(page, "confluence:DEV")

        call_args = mock_add.call_args
        url_to_doc = call_args[0][5]  # 6th positional arg
        assert page.url in url_to_doc
        assert url_to_doc[page.url] == "Full document content."


class TestEmptyContent:
    """Pages with empty/blank content are skipped."""

    @pytest.mark.asyncio
    async def test_empty_string(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="")

        result = await processor.process_page(page, "confluence:DEV")
        assert result.status == PageProcessingStatus.SKIPPED_EMPTY

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="   \n\n  ")

        result = await processor.process_page(page, "confluence:DEV")
        assert result.status == PageProcessingStatus.SKIPPED_EMPTY

    @pytest.mark.asyncio
    async def test_none_content(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content=None)
        page.markdown_content = None

        result = await processor.process_page(page, "confluence:DEV")
        assert result.status == PageProcessingStatus.SKIPPED_EMPTY


class TestMultiPageProcessing:
    """Processing a CrawlResult with multiple pages."""

    @pytest.mark.asyncio
    async def test_processes_all_pages(self):
        pages = [
            _make_page(page_id="1", title="Page 1", content="Content one."),
            _make_page(page_id="2", title="Page 2", content="Content two."),
            _make_page(page_id="3", title="Page 3", content=""),  # empty
        ]
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        crawl = _make_crawl_result(pages=pages)

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_UPDATE_SOURCE), \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"):
            result = await processor.process_crawl_result(crawl)

        assert len(result.page_results) == 3
        assert result.summary.pages_processed == 2
        assert result.summary.pages_skipped_empty == 1


class TestDuplicateDetection:
    """Unchanged pages are skipped; changed pages are re-stored."""

    @pytest.mark.asyncio
    async def test_unchanged_page_skipped(self):
        client = _mock_supabase_client()
        # Simulate DB returning matching last_modified
        select_chain = MagicMock()
        select_chain.execute.return_value = MagicMock(
            data=[{"metadata": {"last_modified": "2025-06-01T00:00:00Z"}}]
        )
        select_chain.eq.return_value = select_chain
        select_chain.limit.return_value = select_chain
        table_mock = MagicMock()
        table_mock.select.return_value = select_chain
        client.table.return_value = table_mock

        processor = ConfluenceProcessor(client)
        page = _make_page(last_modified="2025-06-01T00:00:00Z")

        with patch.dict("os.environ", {"CONFLUENCE_SKIP_UNCHANGED": "true"}):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.status == PageProcessingStatus.SKIPPED_UNCHANGED

    @pytest.mark.asyncio
    async def test_changed_page_stored(self):
        client = _mock_supabase_client()
        # Simulate DB returning different last_modified
        select_chain = MagicMock()
        select_chain.execute.return_value = MagicMock(
            data=[{"metadata": {"last_modified": "2025-05-01T00:00:00Z"}}]
        )
        select_chain.eq.return_value = select_chain
        select_chain.limit.return_value = select_chain
        table_mock = MagicMock()
        table_mock.select.return_value = select_chain
        client.table.return_value = table_mock

        processor = ConfluenceProcessor(client)
        page = _make_page(last_modified="2025-06-01T00:00:00Z")

        with patch(_PATCH_ADD_DOCS), \
             patch.dict("os.environ", {"CONFLUENCE_SKIP_UNCHANGED": "true"}):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.status == PageProcessingStatus.STORED

    @pytest.mark.asyncio
    async def test_skip_unchanged_disabled(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page()

        with patch(_PATCH_ADD_DOCS), \
             patch.dict("os.environ", {"CONFLUENCE_SKIP_UNCHANGED": "false"}):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.status == PageProcessingStatus.STORED


class TestOrphanDeletion:
    """Orphaned chunks (pages removed from Confluence) are deleted."""

    @pytest.mark.asyncio
    async def test_orphans_deleted(self):
        client = _mock_supabase_client()

        # Mock: DB has pages A, B, C; crawl result only has A
        select_chain = MagicMock()
        select_chain.execute.return_value = MagicMock(
            data=[
                {"url": "https://example.com/A"},
                {"url": "https://example.com/B"},
                {"url": "https://example.com/C"},
            ]
        )
        select_chain.eq.return_value = select_chain

        delete_chain = MagicMock()
        delete_chain.eq.return_value = delete_chain
        delete_chain.execute.return_value = MagicMock(data=[])

        table_mock = MagicMock()
        table_mock.select.return_value = select_chain
        table_mock.delete.return_value = delete_chain
        client.table.return_value = table_mock

        processor = ConfluenceProcessor(client)
        page = _make_page(url="https://example.com/A")
        crawl = _make_crawl_result(pages=[page])

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_UPDATE_SOURCE), \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"):
            result = await processor.process_crawl_result(crawl, detect_deletions=True)

        assert result.summary.orphaned_pages_deleted == 2


class TestCodeExamples:
    """Code examples are extracted and stored when USE_AGENTIC_RAG=true."""

    @pytest.mark.asyncio
    async def test_code_examples_stored_when_enabled(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="# Code\n\n```python\nprint('hello')\n```\n\nMore text.")

        code_blocks = [
            {
                "code": "print('hello')",
                "language": "python",
                "context_before": "# Code",
                "context_after": "More text.",
                "full_context": "# Code\n\nprint('hello')\n\nMore text.",
            }
        ]

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_EXTRACT_CODE, return_value=code_blocks), \
             patch(_PATCH_GEN_CODE_SUMMARY, return_value="Prints hello."), \
             patch(_PATCH_ADD_CODE) as mock_add_code, \
             patch.dict("os.environ", {"USE_AGENTIC_RAG": "true"}):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.status == PageProcessingStatus.STORED
        assert result.code_examples_stored == 1
        mock_add_code.assert_called_once()

    @pytest.mark.asyncio
    async def test_code_examples_not_extracted_when_disabled(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="# Code\n\n```python\nprint('hello')\n```")

        with patch(_PATCH_ADD_DOCS), \
             patch(_PATCH_ADD_CODE) as mock_add_code, \
             patch.dict("os.environ", {"USE_AGENTIC_RAG": "false"}):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.code_examples_stored == 0
        mock_add_code.assert_not_called()


class TestSourceUpdateOrder:
    """Source is updated BEFORE documents are inserted (FK constraint)."""

    @pytest.mark.asyncio
    async def test_source_updated_before_documents(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        crawl = _make_crawl_result()

        call_order = []

        def track_update(*args, **kwargs):
            call_order.append("update_source")

        def track_add(*args, **kwargs):
            call_order.append("add_documents")

        with patch(_PATCH_UPDATE_SOURCE, side_effect=track_update), \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"), \
             patch(_PATCH_ADD_DOCS, side_effect=track_add):
            await processor.process_crawl_result(crawl)

        assert call_order.index("update_source") < call_order.index("add_documents")


class TestProcessingFailure:
    """Processing errors are caught and reported as FAILED."""

    @pytest.mark.asyncio
    async def test_failed_page_has_error_message(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="Some content.")

        with patch(_PATCH_ADD_DOCS, side_effect=RuntimeError("DB connection lost")):
            result = await processor.process_page(page, "confluence:DEV")

        assert result.status == PageProcessingStatus.FAILED
        assert "DB connection lost" in result.error

    @pytest.mark.asyncio
    async def test_failed_page_counted_in_summary(self):
        client = _mock_supabase_client()
        processor = ConfluenceProcessor(client)
        page = _make_page(content="Some content.")
        crawl = _make_crawl_result(pages=[page])

        with patch(_PATCH_ADD_DOCS, side_effect=RuntimeError("fail")), \
             patch(_PATCH_UPDATE_SOURCE), \
             patch(_PATCH_EXTRACT_SUMMARY, return_value="summary"):
            result = await processor.process_crawl_result(crawl)

        assert result.summary.pages_failed == 1
        assert result.summary.pages_processed == 0


class TestChunkHelpers:
    """Sanity checks for the copied pure helper functions."""

    def test_smart_chunk_markdown_basic(self):
        text = "A" * 100
        chunks = _smart_chunk_markdown(text, chunk_size=50)
        assert len(chunks) >= 2
        assert "".join(chunks) == text

    def test_extract_section_info(self):
        chunk = "# Heading One\n\nSome text here.\n\n## Heading Two\n\nMore text."
        info = _extract_section_info(chunk)
        assert "Heading One" in info["headers"]
        assert "Heading Two" in info["headers"]
        assert info["char_count"] == len(chunk)
        assert info["word_count"] > 0
