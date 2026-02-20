"""Tests for PATAuthProvider"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.atlassian.auth.pat_auth import PATAuthProvider
from src.atlassian.base import AuthMethod, AtlassianConfigError


def _config(overrides=None):
    cfg = {
        "pat": "my-personal-access-token",
        "base_url": "https://confluence.corp.local",
    }
    if overrides:
        cfg.update(overrides)
    return cfg


class TestValidation:
    def test_valid(self):
        p = PATAuthProvider(_config())
        assert p.pat == "my-personal-access-token"

    def test_missing_pat(self):
        with pytest.raises(AtlassianConfigError, match="pat"):
            PATAuthProvider(_config({"pat": ""}))

    def test_missing_base_url(self):
        with pytest.raises(AtlassianConfigError, match="base_url"):
            PATAuthProvider(_config({"base_url": ""}))


class TestHeaders:
    @pytest.mark.asyncio
    async def test_bearer_header(self):
        p = PATAuthProvider(_config())
        auth = await p.get_auth_headers()
        assert auth.headers["Authorization"] == "Bearer my-personal-access-token"
        assert auth.auth_method == AuthMethod.PAT

    @pytest.mark.asyncio
    async def test_token_always_valid(self):
        p = PATAuthProvider(_config())
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

        with patch("src.atlassian.auth.pat_auth.aiohttp.ClientSession", return_value=mock_session):
            p = PATAuthProvider(_config())
            health = await p.health_check()
            assert health.is_healthy
            assert health.auth_method == "pat"
