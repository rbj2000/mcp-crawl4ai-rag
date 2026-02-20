"""Tests for BasicAuthProvider"""

import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.atlassian.auth.basic_auth import BasicAuthProvider
from src.atlassian.base import AuthMethod, AtlassianConfigError


def _config(overrides=None):
    cfg = {
        "username": "user@example.com",
        "api_token": "my-api-token",
        "base_url": "https://myorg.atlassian.net",
    }
    if overrides:
        cfg.update(overrides)
    return cfg


class TestValidation:
    def test_valid(self):
        p = BasicAuthProvider(_config())
        assert p.username == "user@example.com"

    def test_missing_username(self):
        with pytest.raises(AtlassianConfigError, match="username"):
            BasicAuthProvider(_config({"username": ""}))

    def test_missing_api_token(self):
        with pytest.raises(AtlassianConfigError, match="api_token"):
            BasicAuthProvider(_config({"api_token": ""}))

    def test_missing_base_url(self):
        with pytest.raises(AtlassianConfigError, match="base_url"):
            BasicAuthProvider(_config({"base_url": ""}))


class TestHeaders:
    @pytest.mark.asyncio
    async def test_auth_header_format(self):
        p = BasicAuthProvider(_config())
        auth = await p.get_auth_headers()
        expected = base64.b64encode(b"user@example.com:my-api-token").decode()
        assert auth.headers["Authorization"] == f"Basic {expected}"
        assert auth.auth_method == AuthMethod.BASIC

    @pytest.mark.asyncio
    async def test_token_always_valid(self):
        p = BasicAuthProvider(_config())
        assert await p.is_token_valid() is True


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_healthy(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.atlassian.auth.basic_auth.aiohttp.ClientSession", return_value=mock_session):
            p = BasicAuthProvider(_config())
            health = await p.health_check()
            assert health.is_healthy
            assert health.auth_method == "basic"

    @pytest.mark.asyncio
    async def test_unhealthy(self):
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.text = AsyncMock(return_value="Unauthorized")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.atlassian.auth.basic_auth.aiohttp.ClientSession", return_value=mock_session):
            p = BasicAuthProvider(_config())
            health = await p.health_check()
            assert not health.is_healthy
            assert "401" in health.error_message
