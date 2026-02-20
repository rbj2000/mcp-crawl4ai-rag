"""Confluence content crawler — fetches pages via REST API and converts to Markdown"""

import logging
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse, parse_qs

from .base import AtlassianAPIError, DeploymentType
from .content_converter import convert_content
from .http_client import AtlassianHTTPClient

logger = logging.getLogger(__name__)

DEFAULT_MAX_PAGES = 100
DEFAULT_MAX_DEPTH = 5


@dataclass
class ConfluencePage:
    """A single crawled Confluence page."""

    page_id: str
    title: str
    space_key: str
    url: str
    markdown_content: str
    author: Optional[str] = None
    created: Optional[str] = None
    last_modified: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    parent_page_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrawlSummary:
    """Aggregate stats for a crawl operation."""

    pages_succeeded: int = 0
    pages_failed: int = 0
    page_ids_crawled: Set[str] = field(default_factory=set)
    errors: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class CrawlResult:
    """Result container for a crawl operation."""

    pages: List[ConfluencePage] = field(default_factory=list)
    summary: CrawlSummary = field(default_factory=CrawlSummary)
    space_key: Optional[str] = None


class ConfluenceCrawler:
    """Crawls Confluence pages via REST API and converts content to Markdown.

    Uses ``AtlassianHTTPClient.request()`` for all HTTP communication.
    Cloud uses v2 API; On-Prem / Data Center uses v1 API.
    """

    def __init__(
        self,
        client: AtlassianHTTPClient,
        deployment_type: DeploymentType,
    ):
        self.client = client
        self.deployment_type = deployment_type
        self.max_pages = int(os.getenv("CONFLUENCE_MAX_PAGES", str(DEFAULT_MAX_PAGES)))
        self.max_depth = int(os.getenv("CONFLUENCE_MAX_DEPTH", str(DEFAULT_MAX_DEPTH)))

    @property
    def is_cloud(self) -> bool:
        return self.deployment_type == DeploymentType.CLOUD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl_page(self, page_id: str) -> ConfluencePage:
        """Crawl a single page by ID.

        Raises:
            AtlassianAPIError: on HTTP errors.
        """
        if self.is_cloud:
            return await self._crawl_page_cloud(page_id)
        return await self._crawl_page_onprem(page_id)

    async def crawl_page_by_url(self, url: str) -> ConfluencePage:
        """Parse a Confluence URL and crawl the identified page."""
        page_id = self._parse_page_id_from_url(url)
        return await self.crawl_page(page_id)

    async def crawl_space(self, space_key: str) -> CrawlResult:
        """Crawl all pages in a Confluence space (paginated)."""
        result = CrawlResult(space_key=space_key)

        if self.is_cloud:
            page_ids = await self._list_space_pages_cloud(space_key)
        else:
            page_ids = await self._list_space_pages_onprem(space_key)

        for pid in page_ids[: self.max_pages]:
            if pid in result.summary.page_ids_crawled:
                continue
            await self._safe_crawl_page(pid, result)

        return result

    async def crawl_page_tree(self, root_page_id: str) -> CrawlResult:
        """BFS crawl of a page tree starting from *root_page_id*.

        Respects ``max_depth`` and ``max_pages``. Skips 403/404 pages.
        """
        result = CrawlResult()
        queue: deque = deque()
        queue.append((root_page_id, 0))
        visited: Set[str] = set()

        while queue and len(result.summary.page_ids_crawled) < self.max_pages:
            page_id, depth = queue.popleft()

            if page_id in visited:
                continue
            visited.add(page_id)

            page = await self._safe_crawl_page(page_id, result)
            if page is None:
                continue

            if depth < self.max_depth:
                children = await self._get_child_page_ids(page_id)
                for child_id in children:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))

        return result

    # ------------------------------------------------------------------
    # Cloud (v2) implementation
    # ------------------------------------------------------------------

    async def _crawl_page_cloud(self, page_id: str) -> ConfluencePage:
        data = await self.client.request(
            "GET",
            f"/wiki/api/v2/pages/{page_id}",
            params={"body-format": "atlas_doc_format"},
        )
        return self._parse_cloud_page(data)

    async def _list_space_pages_cloud(self, space_key: str) -> List[str]:
        """List page IDs in a Cloud space using cursor pagination."""
        page_ids: List[str] = []
        # First resolve space ID from space key
        spaces_data = await self.client.request(
            "GET",
            "/wiki/api/v2/spaces",
            params={"keys": space_key, "limit": 1},
        )
        results = spaces_data.get("results", [])
        if not results:
            logger.warning("Space not found: %s", space_key)
            return []
        space_id = results[0]["id"]

        cursor: Optional[str] = None
        while len(page_ids) < self.max_pages:
            params: Dict[str, Any] = {"limit": 25}
            if cursor:
                params["cursor"] = cursor
            data = await self.client.request(
                "GET",
                f"/wiki/api/v2/spaces/{space_id}/pages",
                params=params,
            )
            for page in data.get("results", []):
                page_ids.append(str(page["id"]))

            # Follow cursor
            next_link = data.get("_links", {}).get("next")
            if not next_link:
                break
            cursor = self._extract_cursor(next_link)
            if not cursor:
                break

        return page_ids

    def _parse_cloud_page(self, data: dict) -> ConfluencePage:
        page_id = str(data["id"])
        title = data.get("title", "")
        space_key = data.get("spaceId", "")
        # Extract space key from nested data if available
        if "space" in data:
            space_key = data["space"].get("key", space_key)

        # Extract ADF body
        body_data = data.get("body", {}).get("atlas_doc_format", {})
        adf_value = body_data.get("value")
        if isinstance(adf_value, str):
            import json
            try:
                adf_value = json.loads(adf_value)
            except (json.JSONDecodeError, TypeError):
                adf_value = {}
        markdown = convert_content(adf_value or {}, "adf")

        # URL
        base_link = data.get("_links", {}).get("base", "")
        web_ui = data.get("_links", {}).get("webui", "")
        url = f"{base_link}{web_ui}" if base_link else web_ui

        # Version info
        version = data.get("version", {})
        author = version.get("by", {}).get("displayName") if version else None
        created = data.get("createdAt")
        last_modified = version.get("createdAt") if version else None

        # Labels
        labels = [
            lbl.get("name", "") for lbl in data.get("labels", {}).get("results", [])
        ]

        # Parent
        parent_id = data.get("parentId")

        return ConfluencePage(
            page_id=page_id,
            title=title,
            space_key=str(space_key),
            url=url,
            markdown_content=markdown,
            author=author,
            created=created,
            last_modified=last_modified,
            labels=labels,
            parent_page_id=str(parent_id) if parent_id else None,
        )

    # ------------------------------------------------------------------
    # On-Prem / Data Center (v1) implementation
    # ------------------------------------------------------------------

    async def _crawl_page_onprem(self, page_id: str) -> ConfluencePage:
        data = await self.client.request(
            "GET",
            f"/rest/api/content/{page_id}",
            params={"expand": "body.storage,metadata.labels,ancestors,version,space"},
        )
        return self._parse_onprem_page(data)

    async def _list_space_pages_onprem(self, space_key: str) -> List[str]:
        """List page IDs in an On-Prem space using offset pagination."""
        page_ids: List[str] = []
        start = 0
        limit = 25

        while len(page_ids) < self.max_pages:
            data = await self.client.request(
                "GET",
                "/rest/api/content",
                params={
                    "spaceKey": space_key,
                    "type": "page",
                    "start": start,
                    "limit": limit,
                },
            )
            results = data.get("results", [])
            if not results:
                break
            for page in results:
                page_ids.append(str(page["id"]))

            size = data.get("size", len(results))
            if size < limit:
                break
            start += limit

        return page_ids

    def _parse_onprem_page(self, data: dict) -> ConfluencePage:
        page_id = str(data["id"])
        title = data.get("title", "")

        space_key = data.get("space", {}).get("key", "")

        # Storage format body
        storage_body = data.get("body", {}).get("storage", {}).get("value", "")
        markdown = convert_content(storage_body, "storage")

        # URL
        base_link = data.get("_links", {}).get("base", "")
        web_ui = data.get("_links", {}).get("webui", "")
        url = f"{base_link}{web_ui}" if base_link else web_ui

        # Version info
        version = data.get("version", {})
        author = version.get("by", {}).get("displayName") if version else None
        created = version.get("when") if version else None
        last_modified = created  # v1 API stores version date in "when"

        # Labels
        labels_data = data.get("metadata", {}).get("labels", {})
        labels = [lbl.get("name", "") for lbl in labels_data.get("results", [])]

        # Parent from ancestors
        ancestors = data.get("ancestors", [])
        parent_id = str(ancestors[-1]["id"]) if ancestors else None

        return ConfluencePage(
            page_id=page_id,
            title=title,
            space_key=space_key,
            url=url,
            markdown_content=markdown,
            author=author,
            created=created,
            last_modified=last_modified,
            labels=labels,
            parent_page_id=parent_id,
        )

    # ------------------------------------------------------------------
    # Child page retrieval (for page tree crawl)
    # ------------------------------------------------------------------

    async def _get_child_page_ids(self, page_id: str) -> List[str]:
        try:
            if self.is_cloud:
                data = await self.client.request(
                    "GET",
                    f"/wiki/api/v2/pages/{page_id}/children",
                    params={"limit": 50},
                )
                return [str(p["id"]) for p in data.get("results", [])]
            else:
                data = await self.client.request(
                    "GET",
                    f"/rest/api/content/{page_id}/child/page",
                    params={"limit": 50},
                )
                return [str(p["id"]) for p in data.get("results", [])]
        except AtlassianAPIError as e:
            logger.warning("Failed to get children for page %s: %s", page_id, e)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_crawl_page(
        self, page_id: str, result: CrawlResult
    ) -> Optional[ConfluencePage]:
        """Crawl a page, catching 403/404 and recording errors."""
        try:
            page = await self.crawl_page(page_id)
            result.pages.append(page)
            result.summary.pages_succeeded += 1
            result.summary.page_ids_crawled.add(page_id)
            return page
        except AtlassianAPIError as e:
            status = getattr(e, "status_code", None)
            if status in (403, 404):
                logger.info("Skipping page %s (HTTP %s)", page_id, status)
            else:
                logger.warning("Error crawling page %s: %s", page_id, e)
            result.summary.pages_failed += 1
            result.summary.errors.append(
                {"page_id": page_id, "error": str(e)}
            )
            return None

    def _parse_page_id_from_url(self, url: str) -> str:
        """Extract page ID from various Confluence URL formats.

        Supports:
        - Cloud: ``/wiki/spaces/KEY/pages/12345/Title``
        - On-Prem: ``/pages/viewpage.action?pageId=12345``
        - On-Prem: ``/display/KEY/Title`` (not directly resolvable — raises)
        - Direct ID in path: ``/wiki/api/v2/pages/12345``
        """
        parsed = urlparse(url)
        path = parsed.path

        # pageId query param (On-Prem viewpage.action)
        qs = parse_qs(parsed.query)
        if "pageId" in qs:
            return qs["pageId"][0]

        # Cloud: /wiki/spaces/KEY/pages/{id}/...
        match = re.search(r"/pages/(\d+)", path)
        if match:
            return match.group(1)

        # API path: /wiki/api/v2/pages/{id}
        match = re.search(r"/api/v[12]/pages/(\d+)", path)
        if match:
            return match.group(1)

        # On-Prem: /rest/api/content/{id}
        match = re.search(r"/rest/api/content/(\d+)", path)
        if match:
            return match.group(1)

        raise ValueError(f"Cannot extract page ID from URL: {url}")

    @staticmethod
    def _extract_cursor(next_link: str) -> Optional[str]:
        """Extract cursor value from a v2 pagination ``next`` link."""
        match = re.search(r"[?&]cursor=([^&]+)", next_link)
        return match.group(1) if match else None
