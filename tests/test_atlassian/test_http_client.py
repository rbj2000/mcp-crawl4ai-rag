"""Tests for AtlassianHTTPClient — retry, rate-limit, auth injection"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.atlassian.base import (
    AuthHeaders,
    AuthMethod,
    AtlassianAPIError,
    AtlassianAuthError,
    AtlassianRateLimitError,
)
from src.atlassian.http_client import AtlassianHTTPClient


def _mock_auth_provider():
    provider = AsyncMock()
    provider.get_auth_headers = AsyncMock(return_value=AuthHeaders(
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json", "Accept": "application/json"},
        auth_method=AuthMethod.PAT,
    ))
    provider.initialize = AsyncMock()
    provider.close = AsyncMock()
    provider.refresh_if_needed = AsyncMock()
    return provider


def _make_response(status, body=None, content_type="application/json", headers=None):
    resp = AsyncMock()
    resp.status = status
    resp.content_type = content_type
    resp.headers = headers or {}
    if isinstance(body, dict):
        resp.json = AsyncMock(return_value=body)
    else:
        resp.text = AsyncMock(return_value=body or "")
        resp.json = AsyncMock(return_value={})
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


class TestSuccessfulRequests:
    @pytest.mark.asyncio
    async def test_get_json(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=0)
        resp = _make_response(200, {"ok": True})

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=resp)
        client._session = mock_session

        result = await client.request("GET", "/api/test")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_non_json_response(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=0)
        resp = _make_response(200, "plain text", content_type="text/plain")

        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=resp)
        client._session = mock_session

        result = await client.request("GET", "/api/text")
        assert result["_raw"] == "plain text"


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=2)

        fail_resp = _make_response(500, "server error")
        ok_resp = _make_response(200, {"ok": True})

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return fail_resp if call_count <= 1 else ok_resp

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=side_effect)
        client._session = mock_session

        with patch("src.atlassian.http_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.request("GET", "/api/test")
        assert result == {"ok": True}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=1)

        fail_resp = _make_response(502, "bad gateway")
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=fail_resp)
        client._session = mock_session

        with patch("src.atlassian.http_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(AtlassianAPIError, match="502"):
                await client.request("GET", "/api/fail")


class TestAuthRefresh:
    @pytest.mark.asyncio
    async def test_401_triggers_refresh_and_retry(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=2)

        unauth_resp = _make_response(401, "unauthorized")
        ok_resp = _make_response(200, {"ok": True})

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return unauth_resp if call_count <= 1 else ok_resp

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=side_effect)
        client._session = mock_session

        result = await client.request("GET", "/api/test")
        assert result == {"ok": True}
        auth.refresh_if_needed.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_401_raises(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=2)

        unauth_resp = _make_response(401, "unauthorized")
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=unauth_resp)
        client._session = mock_session

        with pytest.raises(AtlassianAuthError, match="Authentication failed"):
            await client.request("GET", "/api/test")


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_429_respects_retry_after(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=2)

        rate_resp = _make_response(429, "rate limited", headers={"Retry-After": "1"})
        ok_resp = _make_response(200, {"ok": True})

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return rate_resp if call_count <= 1 else ok_resp

        mock_session = MagicMock()
        mock_session.request = MagicMock(side_effect=side_effect)
        client._session = mock_session

        with patch("src.atlassian.http_client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.request("GET", "/api/test")
        assert result == {"ok": True}
        mock_sleep.assert_called_once_with(1.0)

    @pytest.mark.asyncio
    async def test_429_raises_after_retries(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com", max_retries=0)

        rate_resp = _make_response(429, "rate limited")
        mock_session = MagicMock()
        mock_session.request = MagicMock(return_value=rate_resp)
        client._session = mock_session

        with pytest.raises(AtlassianRateLimitError):
            await client.request("GET", "/api/test")


class TestNotInitialized:
    @pytest.mark.asyncio
    async def test_raises_if_not_initialized(self):
        auth = _mock_auth_provider()
        client = AtlassianHTTPClient(auth, "https://example.com")
        with pytest.raises(RuntimeError, match="not initialized"):
            await client.request("GET", "/")
