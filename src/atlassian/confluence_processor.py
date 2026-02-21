"""Confluence content processing — chunks, embeds, and stores crawled pages in the vector DB."""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .confluence_crawler import ConfluencePage, CrawlResult

# Import utils module (not individual functions) so tests can patch via src.utils.*
import src.utils as _utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class PageProcessingStatus(Enum):
    STORED = "stored"
    SKIPPED_UNCHANGED = "skipped_unchanged"
    SKIPPED_EMPTY = "skipped_empty"
    FAILED = "failed"


@dataclass
class PageResult:
    page_id: str
    title: str
    url: str
    status: PageProcessingStatus
    chunks_stored: int = 0
    code_examples_stored: int = 0
    error: Optional[str] = None


@dataclass
class ProcessingSummary:
    pages_processed: int = 0
    pages_skipped_unchanged: int = 0
    pages_skipped_empty: int = 0
    pages_failed: int = 0
    total_chunks_stored: int = 0
    total_code_examples_stored: int = 0
    orphaned_pages_deleted: int = 0


@dataclass
class ProcessingResult:
    page_results: List[PageResult] = field(default_factory=list)
    summary: ProcessingSummary = field(default_factory=ProcessingSummary)
    source_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Pure helpers (copied from crawl4ai_mcp.py to avoid heavy side-effect imports)
# ---------------------------------------------------------------------------

def _smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks


def _extract_section_info(chunk: str) -> Dict[str, Any]:
    """Extract headers and stats from a chunk."""
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class ConfluenceProcessor:
    """Processes crawled Confluence pages: chunks, embeds, and stores in the vector DB.

    Uses the legacy ``utils.py`` storage functions (``add_documents_to_supabase``,
    ``update_source_info``, etc.) which are currently wired into the MCP server.
    """

    def __init__(self, supabase_client: Any) -> None:
        self._client = supabase_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_crawl_result(
        self,
        crawl_result: CrawlResult,
        *,
        detect_deletions: bool = False,
    ) -> ProcessingResult:
        """Process all pages from a :class:`CrawlResult` and store them.

        Args:
            crawl_result: Result from :class:`ConfluenceCrawler`.
            detect_deletions: If *True*, remove orphaned chunks from the DB
                whose URLs are no longer present in the crawl result.
        """
        space_key = crawl_result.space_key or self._infer_space_key(crawl_result)
        source_id = f"confluence:{space_key}"

        result = ProcessingResult(source_id=source_id)

        # FK constraint: source row must exist before document rows.
        await self._update_source(source_id, crawl_result.pages)

        for page in crawl_result.pages:
            page_result = await self.process_page(page, source_id)
            result.page_results.append(page_result)

            # Aggregate summary
            if page_result.status == PageProcessingStatus.STORED:
                result.summary.pages_processed += 1
                result.summary.total_chunks_stored += page_result.chunks_stored
                result.summary.total_code_examples_stored += page_result.code_examples_stored
            elif page_result.status == PageProcessingStatus.SKIPPED_UNCHANGED:
                result.summary.pages_skipped_unchanged += 1
            elif page_result.status == PageProcessingStatus.SKIPPED_EMPTY:
                result.summary.pages_skipped_empty += 1
            elif page_result.status == PageProcessingStatus.FAILED:
                result.summary.pages_failed += 1

        if detect_deletions:
            current_urls = {p.url for p in crawl_result.pages}
            deleted = await self._detect_and_delete_orphans(source_id, current_urls)
            result.summary.orphaned_pages_deleted = deleted

        return result

    async def process_page(self, page: ConfluencePage, source_id: str) -> PageResult:
        """Process a single Confluence page and store its chunks."""

        # Guard: empty content
        if not page.markdown_content or not page.markdown_content.strip():
            return PageResult(
                page_id=page.page_id,
                title=page.title,
                url=page.url,
                status=PageProcessingStatus.SKIPPED_EMPTY,
            )

        # Duplicate check
        if await self._is_page_unchanged(page, source_id):
            return PageResult(
                page_id=page.page_id,
                title=page.title,
                url=page.url,
                status=PageProcessingStatus.SKIPPED_UNCHANGED,
            )

        try:
            chunks = _smart_chunk_markdown(page.markdown_content)
            if not chunks:
                return PageResult(
                    page_id=page.page_id,
                    title=page.title,
                    url=page.url,
                    status=PageProcessingStatus.SKIPPED_EMPTY,
                )

            # Build parallel lists expected by add_documents_to_supabase
            urls: List[str] = []
            chunk_numbers: List[int] = []
            contents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            total_word_count = 0

            for i, chunk in enumerate(chunks):
                urls.append(page.url)
                chunk_numbers.append(i)
                contents.append(chunk)

                section = _extract_section_info(chunk)
                meta: Dict[str, Any] = {
                    # Standard fields (matching crawl4ai_mcp.py pattern)
                    "chunk_index": i,
                    "url": page.url,
                    "source": source_id,
                    "headers": section["headers"],
                    "char_count": section["char_count"],
                    "word_count": section["word_count"],
                    # Confluence-specific fields
                    "source_type": "confluence",
                    "space_key": page.space_key,
                    "page_id": page.page_id,
                    "page_title": page.title,
                    "labels": page.labels,
                    "author": page.author,
                    "last_modified": page.last_modified,
                    "parent_page_id": page.parent_page_id,
                }
                metadatas.append(meta)
                total_word_count += section["word_count"]

            url_to_full_document = {page.url: page.markdown_content}

            # Store document chunks
            await self._run_sync(
                _utils.add_documents_to_supabase,
                self._client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )

            # Code examples (agentic RAG)
            code_examples_stored = 0
            if os.getenv("USE_AGENTIC_RAG", "false") == "true":
                code_examples_stored = await self._process_code_examples(
                    page, source_id
                )

            return PageResult(
                page_id=page.page_id,
                title=page.title,
                url=page.url,
                status=PageProcessingStatus.STORED,
                chunks_stored=len(chunks),
                code_examples_stored=code_examples_stored,
            )

        except Exception as exc:
            logger.error("Failed to process page %s: %s", page.page_id, exc)
            return PageResult(
                page_id=page.page_id,
                title=page.title,
                url=page.url,
                status=PageProcessingStatus.FAILED,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _process_code_examples(
        self, page: ConfluencePage, source_id: str
    ) -> int:
        """Extract and store code examples from a page. Returns count stored."""
        code_blocks = _utils.extract_code_blocks(page.markdown_content)
        if not code_blocks:
            return 0

        code_urls: List[str] = []
        code_chunk_numbers: List[int] = []
        code_examples: List[str] = []
        code_summaries: List[str] = []
        code_metadatas: List[Dict[str, Any]] = []

        for i, block in enumerate(code_blocks):
            summary = await self._run_sync(
                _utils.generate_code_example_summary,
                block["code"],
                block["context_before"],
                block["context_after"],
            )
            code_urls.append(page.url)
            code_chunk_numbers.append(i)
            code_examples.append(block["code"])
            code_summaries.append(summary)
            code_metadatas.append(
                {
                    "chunk_index": i,
                    "url": page.url,
                    "source": source_id,
                    "source_type": "confluence",
                    "space_key": page.space_key,
                    "page_id": page.page_id,
                    "char_count": len(block["code"]),
                    "word_count": len(block["code"].split()),
                }
            )

        await self._run_sync(
            _utils.add_code_examples_to_supabase,
            self._client,
            code_urls,
            code_chunk_numbers,
            code_examples,
            code_summaries,
            code_metadatas,
        )

        return len(code_blocks)

    async def _is_page_unchanged(
        self, page: ConfluencePage, source_id: str
    ) -> bool:
        """Return True if the page already exists in the DB with the same last_modified."""
        if os.getenv("CONFLUENCE_SKIP_UNCHANGED", "true") != "true":
            return False

        if not page.last_modified:
            return False

        try:
            result = await self._run_sync(
                lambda: (
                    self._client.table("crawled_pages")
                    .select("metadata")
                    .eq("url", page.url)
                    .eq("source_id", source_id)
                    .limit(1)
                    .execute()
                ),
            )
            if not result.data:
                return False

            stored_meta = result.data[0].get("metadata", {})
            return stored_meta.get("last_modified") == page.last_modified

        except Exception as exc:
            logger.debug("Duplicate check failed for %s: %s", page.url, exc)
            return False

    async def _detect_and_delete_orphans(
        self, source_id: str, current_page_urls: set
    ) -> int:
        """Delete chunks whose URLs are no longer present in the crawl result."""
        try:
            result = await self._run_sync(
                lambda: (
                    self._client.table("crawled_pages")
                    .select("url")
                    .eq("source_id", source_id)
                    .execute()
                ),
            )

            stored_urls = {row["url"] for row in result.data} if result.data else set()
            orphan_urls = stored_urls - current_page_urls

            for url in orphan_urls:
                await self._run_sync(
                    lambda u=url: (
                        self._client.table("crawled_pages")
                        .delete()
                        .eq("url", u)
                        .eq("source_id", source_id)
                        .execute()
                    ),
                )
                # Also clean code_examples
                try:
                    await self._run_sync(
                        lambda u=url: (
                            self._client.table("code_examples")
                            .delete()
                            .eq("url", u)
                            .execute()
                        ),
                    )
                except Exception:
                    pass  # code_examples table may not exist

            return len(orphan_urls)

        except Exception as exc:
            logger.error("Orphan detection failed for %s: %s", source_id, exc)
            return 0

    async def _update_source(
        self, source_id: str, pages: List[ConfluencePage]
    ) -> None:
        """Create/update the source row (FK constraint: must exist before documents)."""
        combined_content = "\n\n".join(
            p.markdown_content[:2000] for p in pages if p.markdown_content
        )
        total_words = sum(
            len(p.markdown_content.split()) for p in pages if p.markdown_content
        )

        summary = await self._run_sync(
            _utils.extract_source_summary, source_id, combined_content[:5000]
        )
        await self._run_sync(
            _utils.update_source_info, self._client, source_id, summary, total_words
        )

    @staticmethod
    async def _run_sync(func, *args, **kwargs):
        """Run a synchronous function in an executor to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        if args or kwargs:
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        return await loop.run_in_executor(None, func)

    @staticmethod
    def _infer_space_key(crawl_result: CrawlResult) -> str:
        """Best-effort space key from the first page when CrawlResult.space_key is None."""
        if crawl_result.pages:
            return crawl_result.pages[0].space_key
        return "unknown"
