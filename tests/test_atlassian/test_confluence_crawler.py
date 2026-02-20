"""Tests for ConfluenceCrawler — mocked at client.request() level"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.atlassian.base import AtlassianAPIError, DeploymentType
from src.atlassian.confluence_crawler import (
    ConfluenceCrawler,
    ConfluencePage,
    CrawlResult,
    CrawlSummary,
)


# ------------------------------------------------------------------
# Fixtures / helpers
# ------------------------------------------------------------------

def _cloud_page_response(page_id="123", title="Test Page", space_key="DEV"):
    """Minimal Cloud v2 page response."""
    return {
        "id": page_id,
        "title": title,
        "spaceId": "space-1",
        "space": {"key": space_key},
        "parentId": "100",
        "createdAt": "2025-01-01T00:00:00Z",
        "body": {
            "atlas_doc_format": {
                "value": json.dumps({
                    "type": "doc",
                    "content": [
                        {"type": "paragraph", "content": [
                            {"type": "text", "text": "Hello from Cloud"}
                        ]}
                    ],
                })
            }
        },
        "version": {
            "by": {"displayName": "Alice"},
            "createdAt": "2025-01-02T00:00:00Z",
        },
        "labels": {"results": [{"name": "docs"}, {"name": "v2"}]},
        "_links": {"base": "https://acme.atlassian.net/wiki", "webui": "/spaces/DEV/pages/123"},
    }


def _onprem_page_response(page_id="456", title="On-Prem Page", space_key="ENG"):
    """Minimal On-Prem v1 content response."""
    return {
        "id": page_id,
        "title": title,
        "space": {"key": space_key},
        "body": {
            "storage": {
                "value": "<p>Hello from On-Prem</p>",
            }
        },
        "version": {
            "by": {"displayName": "Bob"},
            "when": "2025-03-01T00:00:00Z",
        },
        "metadata": {
            "labels": {"results": [{"name": "internal"}]},
        },
        "ancestors": [{"id": "10"}, {"id": "50"}],
        "_links": {"base": "https://confluence.corp.com", "webui": "/display/ENG/On-Prem+Page"},
    }


def _make_crawler(deployment_type=DeploymentType.CLOUD, max_pages=100, max_depth=5):
    client = AsyncMock()
    crawler = ConfluenceCrawler(client, deployment_type)
    crawler.max_pages = max_pages
    crawler.max_depth = max_depth
    return crawler, client


# ------------------------------------------------------------------
# Single page crawl
# ------------------------------------------------------------------

class TestCrawlPageCloud:
    @pytest.mark.asyncio
    async def test_crawl_page_cloud(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)
        client.request.return_value = _cloud_page_response()

        page = await crawler.crawl_page("123")

        assert isinstance(page, ConfluencePage)
        assert page.page_id == "123"
        assert page.title == "Test Page"
        assert page.space_key == "DEV"
        assert "Hello from Cloud" in page.markdown_content
        assert page.author == "Alice"
        assert page.labels == ["docs", "v2"]
        assert page.parent_page_id == "100"
        assert "atlassian.net" in page.url

        client.request.assert_called_once_with(
            "GET",
            "/wiki/api/v2/pages/123",
            params={"body-format": "atlas_doc_format"},
        )


class TestCrawlPageOnPrem:
    @pytest.mark.asyncio
    async def test_crawl_page_onprem(self):
        crawler, client = _make_crawler(DeploymentType.ON_PREM)
        client.request.return_value = _onprem_page_response()

        page = await crawler.crawl_page("456")

        assert page.page_id == "456"
        assert page.title == "On-Prem Page"
        assert page.space_key == "ENG"
        assert "Hello from On-Prem" in page.markdown_content
        assert page.author == "Bob"
        assert page.labels == ["internal"]
        assert page.parent_page_id == "50"  # last ancestor

        client.request.assert_called_once_with(
            "GET",
            "/rest/api/content/456",
            params={"expand": "body.storage,metadata.labels,ancestors,version,space"},
        )


# ------------------------------------------------------------------
# Crawl page by URL
# ------------------------------------------------------------------

class TestCrawlPageByURL:
    @pytest.mark.asyncio
    async def test_cloud_url(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)
        client.request.return_value = _cloud_page_response()

        page = await crawler.crawl_page_by_url(
            "https://acme.atlassian.net/wiki/spaces/DEV/pages/123/Test-Page"
        )
        assert page.page_id == "123"

    @pytest.mark.asyncio
    async def test_onprem_viewpage_url(self):
        crawler, client = _make_crawler(DeploymentType.ON_PREM)
        client.request.return_value = _onprem_page_response()

        page = await crawler.crawl_page_by_url(
            "https://confluence.corp.com/pages/viewpage.action?pageId=456"
        )
        assert page.page_id == "456"

    @pytest.mark.asyncio
    async def test_invalid_url_raises(self):
        crawler, client = _make_crawler()
        with pytest.raises(ValueError, match="Cannot extract page ID"):
            await crawler.crawl_page_by_url("https://example.com/no-page-id-here")


# ------------------------------------------------------------------
# Space crawl
# ------------------------------------------------------------------

class TestCrawlSpaceCloud:
    @pytest.mark.asyncio
    async def test_crawl_space_single_page(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)

        async def mock_request(method, path, **kwargs):
            if "/v2/spaces" == path.rstrip("/") or path == "/wiki/api/v2/spaces":
                return {"results": [{"id": "space-1", "key": "DEV"}]}
            if "space-1/pages" in path:
                return {
                    "results": [{"id": "123"}],
                    "_links": {},
                }
            # Page fetch
            return _cloud_page_response()

        client.request.side_effect = mock_request
        result = await crawler.crawl_space("DEV")

        assert isinstance(result, CrawlResult)
        assert result.space_key == "DEV"
        assert result.summary.pages_succeeded == 1
        assert result.summary.pages_failed == 0
        assert len(result.pages) == 1

    @pytest.mark.asyncio
    async def test_crawl_space_cursor_pagination(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)

        call_count = 0

        async def mock_request(method, path, **kwargs):
            nonlocal call_count
            if "/v2/spaces" == path.rstrip("/") or path == "/wiki/api/v2/spaces":
                return {"results": [{"id": "space-1", "key": "DEV"}]}
            if "space-1/pages" in path:
                call_count += 1
                if call_count == 1:
                    return {
                        "results": [{"id": "1"}],
                        "_links": {"next": "/wiki/api/v2/spaces/space-1/pages?cursor=abc123"},
                    }
                return {
                    "results": [{"id": "2"}],
                    "_links": {},
                }
            return _cloud_page_response(page_id=path.split("/")[-1])

        client.request.side_effect = mock_request
        result = await crawler.crawl_space("DEV")
        assert result.summary.pages_succeeded == 2


class TestCrawlSpaceOnPrem:
    @pytest.mark.asyncio
    async def test_crawl_space_offset_pagination(self):
        crawler, client = _make_crawler(DeploymentType.ON_PREM)

        call_count = 0

        async def mock_request(method, path, **kwargs):
            nonlocal call_count
            params = kwargs.get("params", {})
            if path == "/rest/api/content" and params.get("type") == "page":
                call_count += 1
                if call_count == 1:
                    # Return size == limit to trigger next page
                    return {
                        "results": [{"id": str(i)} for i in range(10, 35)],
                        "size": 25,
                    }
                # Second call — fewer than limit means last page
                return {
                    "results": [{"id": "35"}],
                    "size": 1,
                }
            return _onprem_page_response(page_id=path.split("/")[-1])

        client.request.side_effect = mock_request
        result = await crawler.crawl_space("ENG")
        assert result.summary.pages_succeeded == 26
        assert call_count == 2  # two pagination requests


# ------------------------------------------------------------------
# Page tree crawl
# ------------------------------------------------------------------

class TestCrawlPageTree:
    @pytest.mark.asyncio
    async def test_bfs_traversal(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD, max_depth=2)

        async def mock_request(method, path, **kwargs):
            # Page fetch
            if "/wiki/api/v2/pages/" in path and "/children" not in path:
                pid = path.split("/")[-1]
                return _cloud_page_response(page_id=pid, title=f"Page {pid}")
            # Children
            if "/children" in path:
                pid = path.split("/")[-2]
                if pid == "1":
                    return {"results": [{"id": "2"}, {"id": "3"}]}
                if pid == "2":
                    return {"results": [{"id": "4"}]}
                return {"results": []}
            return {}

        client.request.side_effect = mock_request
        result = await crawler.crawl_page_tree("1")

        crawled_ids = {p.page_id for p in result.pages}
        assert "1" in crawled_ids
        assert "2" in crawled_ids
        assert "3" in crawled_ids
        assert "4" in crawled_ids
        assert result.summary.pages_succeeded == 4

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        """Cycles in the page tree should not cause infinite loops."""
        crawler, client = _make_crawler(DeploymentType.CLOUD, max_depth=5)

        async def mock_request(method, path, **kwargs):
            if "/children" in path:
                pid = path.split("/")[-2]
                if pid == "1":
                    return {"results": [{"id": "2"}]}
                if pid == "2":
                    return {"results": [{"id": "1"}]}  # cycle!
                return {"results": []}
            pid = path.split("/")[-1]
            return _cloud_page_response(page_id=pid)

        client.request.side_effect = mock_request
        result = await crawler.crawl_page_tree("1")

        assert result.summary.pages_succeeded == 2  # only 1 and 2
        assert len(result.pages) == 2

    @pytest.mark.asyncio
    async def test_max_depth_limit(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD, max_depth=1)

        async def mock_request(method, path, **kwargs):
            if "/children" in path:
                pid = path.split("/")[-2]
                if pid == "1":
                    return {"results": [{"id": "2"}]}
                if pid == "2":
                    return {"results": [{"id": "3"}]}
                return {"results": []}
            pid = path.split("/")[-1]
            return _cloud_page_response(page_id=pid)

        client.request.side_effect = mock_request
        result = await crawler.crawl_page_tree("1")

        crawled_ids = {p.page_id for p in result.pages}
        assert "1" in crawled_ids
        assert "2" in crawled_ids
        assert "3" not in crawled_ids  # beyond max_depth=1

    @pytest.mark.asyncio
    async def test_max_pages_limit(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD, max_pages=2, max_depth=5)

        async def mock_request(method, path, **kwargs):
            if "/children" in path:
                return {"results": [{"id": "99"}]}
            pid = path.split("/")[-1]
            return _cloud_page_response(page_id=pid)

        client.request.side_effect = mock_request
        result = await crawler.crawl_page_tree("1")

        assert result.summary.pages_succeeded <= 2


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_403_skipped(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)

        async def mock_request(method, path, **kwargs):
            if "forbidden" in path or path.endswith("/2"):
                raise AtlassianAPIError("Forbidden", status_code=403)
            pid = path.split("/")[-1]
            return _cloud_page_response(page_id=pid)

        client.request.side_effect = mock_request
        result = CrawlResult()
        page = await crawler._safe_crawl_page("2", result)

        assert page is None
        assert result.summary.pages_failed == 1
        assert len(result.summary.errors) == 1

    @pytest.mark.asyncio
    async def test_404_skipped(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)
        client.request.side_effect = AtlassianAPIError("Not Found", status_code=404)

        result = CrawlResult()
        page = await crawler._safe_crawl_page("999", result)

        assert page is None
        assert result.summary.pages_failed == 1

    @pytest.mark.asyncio
    async def test_500_recorded_as_error(self):
        crawler, client = _make_crawler(DeploymentType.CLOUD)
        client.request.side_effect = AtlassianAPIError("Server Error", status_code=500)

        result = CrawlResult()
        page = await crawler._safe_crawl_page("500", result)

        assert page is None
        assert result.summary.pages_failed == 1
        assert "Server Error" in result.summary.errors[0]["error"]


# ------------------------------------------------------------------
# URL parsing
# ------------------------------------------------------------------

class TestURLParsing:
    def test_cloud_page_url(self):
        crawler, _ = _make_crawler()
        pid = crawler._parse_page_id_from_url(
            "https://acme.atlassian.net/wiki/spaces/DEV/pages/12345/My-Page"
        )
        assert pid == "12345"

    def test_onprem_viewpage_url(self):
        crawler, _ = _make_crawler()
        pid = crawler._parse_page_id_from_url(
            "https://confluence.corp.com/pages/viewpage.action?pageId=67890"
        )
        assert pid == "67890"

    def test_rest_api_url(self):
        crawler, _ = _make_crawler()
        pid = crawler._parse_page_id_from_url(
            "https://confluence.corp.com/rest/api/content/111"
        )
        assert pid == "111"

    def test_v2_api_url(self):
        crawler, _ = _make_crawler()
        pid = crawler._parse_page_id_from_url(
            "https://acme.atlassian.net/wiki/api/v2/pages/222"
        )
        assert pid == "222"

    def test_unparseable_url_raises(self):
        crawler, _ = _make_crawler()
        with pytest.raises(ValueError):
            crawler._parse_page_id_from_url("https://example.com/random/path")


# ------------------------------------------------------------------
# Dataclass basic tests
# ------------------------------------------------------------------

class TestDataclasses:
    def test_confluence_page_defaults(self):
        page = ConfluencePage(
            page_id="1", title="T", space_key="S", url="u", markdown_content="m"
        )
        assert page.labels == []
        assert page.metadata == {}
        assert page.parent_page_id is None

    def test_crawl_summary_defaults(self):
        s = CrawlSummary()
        assert s.pages_succeeded == 0
        assert s.pages_failed == 0
        assert s.page_ids_crawled == set()
        assert s.errors == []

    def test_crawl_result_defaults(self):
        r = CrawlResult()
        assert r.pages == []
        assert r.space_key is None
