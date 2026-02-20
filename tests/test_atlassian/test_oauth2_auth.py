"""Tests for OAuth2AuthProvider"""

import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.atlassian.auth.oauth2_auth import OAuth2AuthProvider, EXPIRY_BUFFER_SECONDS
from src.atlassian.base import AuthMethod, AtlassianConfigError, AtlassianAuthError


def _config(overrides=None):
    cfg = {
        "oauth2_client_id": "cid",
        "oauth2_client_secret": "csecret",
        "base_url": "https://myorg.atlassian.net",
        "oauth2_access_token": "access-tok",
        "oauth2_refresh_token": "refresh-tok",
        "oauth2_token_expires_at": str(time.time() + 7200),
    }
    if overrides:
        cfg.update(overrides)
    return cfg


class TestValidation:
    def test_valid(self):
        p = OAuth2AuthProvider(_config())
        assert p.client_id == "cid"

    def test_missing_client_id(self):
        with pytest.raises(AtlassianConfigError, match="oauth2_client_id"):
            OAuth2AuthProvider(_config({"oauth2_client_id": ""}))

    def test_missing_tokens(self):
        with pytest.raises(AtlassianConfigError, match="oauth2_access_token or oauth2_refresh_token"):
            OAuth2AuthProvider(_config({
                "oauth2_access_token": None,
                "oauth2_refresh_token": None,
            }))

    def test_refresh_token_only_is_ok(self):
        p = OAuth2AuthProvider(_config({"oauth2_access_token": None}))
        assert p._refresh_token == "refresh-tok"


class TestHeaders:
    @pytest.mark.asyncio
    async def test_bearer_header(self):
        p = OAuth2AuthProvider(_config())
        auth = await p.get_auth_headers()
        assert auth.headers["Authorization"] == "Bearer access-tok"
        assert auth.auth_method == AuthMethod.OAUTH2
        assert auth.expires_at is not None


class TestTokenValidity:
    @pytest.mark.asyncio
    async def test_valid_token(self):
        p = OAuth2AuthProvider(_config())
        assert await p.is_token_valid() is True

    @pytest.mark.asyncio
    async def test_expired_token(self):
        p = OAuth2AuthProvider(_config({"oauth2_token_expires_at": str(time.time() - 100)}))
        assert await p.is_token_valid() is False

    @pytest.mark.asyncio
    async def test_no_expiry_assumed_valid(self):
        p = OAuth2AuthProvider(_config())
        p._expires_at = None
        assert await p.is_token_valid() is True


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_when_near_expiry(self):
        cfg = _config({"oauth2_token_expires_at": str(time.time() + 60)})  # within buffer
        p = OAuth2AuthProvider(cfg)

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        })
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.atlassian.auth.oauth2_auth.aiohttp.ClientSession", return_value=mock_session):
            await p.refresh_if_needed()
            assert p._access_token == "new-access"
            assert p._refresh_token == "new-refresh"

    @pytest.mark.asyncio
    async def test_no_refresh_when_far_from_expiry(self):
        cfg = _config({"oauth2_token_expires_at": str(time.time() + 7200)})
        p = OAuth2AuthProvider(cfg)
        original_token = p._access_token
        await p.refresh_if_needed()
        assert p._access_token == original_token

    @pytest.mark.asyncio
    async def test_refresh_failure_raises(self):
        cfg = _config({"oauth2_token_expires_at": str(time.time() + 60)})
        p = OAuth2AuthProvider(cfg)

        mock_resp = AsyncMock()
        mock_resp.status = 400
        mock_resp.text = AsyncMock(return_value="bad request")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("src.atlassian.auth.oauth2_auth.aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(AtlassianAuthError, match="Token refresh failed"):
                await p.refresh_if_needed()
