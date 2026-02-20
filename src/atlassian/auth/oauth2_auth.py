"""OAuth 2.0 (3-legged) authentication for Atlassian Cloud Enterprise"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

import aiohttp

from ..base import (
    AuthHeaders,
    AuthHealthStatus,
    AuthMethod,
    AtlassianAuthError,
    AtlassianConfigError,
)
from .base_auth import AtlassianAuthProvider

logger = logging.getLogger(__name__)

TOKEN_URL = "https://auth.atlassian.com/oauth/token"
EXPIRY_BUFFER_SECONDS = 300  # refresh 5 min before expiry


class OAuth2AuthProvider(AtlassianAuthProvider):
    """OAuth 2.0 3LO for Atlassian Cloud.

    Tokens are expected to be pre-cached via env vars (headless operation) or
    obtained through the ``scripts/confluence_oauth_setup.py`` browser flow.
    Automatic refresh is performed via ``refresh_token`` grant.
    """

    REQUIRED_SCOPES = [
        "read:confluence-content.all",
        "read:confluence-space.summary",
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client_id: str = config["oauth2_client_id"]
        self.client_secret: str = config["oauth2_client_secret"]
        self.base_url: str = config["base_url"]
        self.cloud_id: Optional[str] = config.get("cloud_id")

        # Tokens
        self._access_token: Optional[str] = config.get("oauth2_access_token")
        self._refresh_token: Optional[str] = config.get("oauth2_refresh_token")
        self._expires_at: Optional[float] = None

        if self._access_token and config.get("oauth2_token_expires_at"):
            self._expires_at = float(config["oauth2_token_expires_at"])

        # Concurrency guard for refresh
        self._refresh_lock = asyncio.Lock()

    def _validate_config(self) -> None:
        missing = []
        for key in ("oauth2_client_id", "oauth2_client_secret", "base_url"):
            if not self.config.get(key):
                missing.append(key)
        if missing:
            raise AtlassianConfigError(
                f"OAuth2AuthProvider missing required config: {missing}"
            )
        # Must have either access_token or refresh_token to function
        if not self.config.get("oauth2_access_token") and not self.config.get("oauth2_refresh_token"):
            raise AtlassianConfigError(
                "OAuth2AuthProvider requires at least one of "
                "oauth2_access_token or oauth2_refresh_token"
            )

    async def get_auth_headers(self) -> AuthHeaders:
        await self.refresh_if_needed()
        if not self._access_token:
            raise AtlassianAuthError("No access token available")
        return AuthHeaders(
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            auth_method=AuthMethod.OAUTH2,
            expires_at=self._expires_at,
        )

    async def is_token_valid(self) -> bool:
        if not self._access_token:
            return False
        if self._expires_at is None:
            return True  # Unknown expiry — assume valid
        return time.time() < self._expires_at

    async def refresh_if_needed(self) -> None:
        """Proactively refresh the access token if it expires within the buffer window."""
        if await self.is_token_valid():
            # Still valid and not close to expiry
            if self._expires_at is None or (self._expires_at - time.time()) > EXPIRY_BUFFER_SECONDS:
                return

        if not self._refresh_token:
            logger.warning("Access token expired and no refresh token available")
            return

        async with self._refresh_lock:
            # Double-check after acquiring lock (another coroutine may have refreshed)
            if self._expires_at and (self._expires_at - time.time()) > EXPIRY_BUFFER_SECONDS:
                return
            await self._do_refresh()

    async def _do_refresh(self) -> None:
        logger.info("Refreshing OAuth2 access token")
        payload = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self._refresh_token,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(TOKEN_URL, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise AtlassianAuthError(
                            f"Token refresh failed (HTTP {resp.status}): {body[:300]}"
                        )
                    data = await resp.json()

            self._access_token = data["access_token"]
            if "refresh_token" in data:
                self._refresh_token = data["refresh_token"]
            expires_in = data.get("expires_in", 3600)
            self._expires_at = time.time() + expires_in
            logger.info("OAuth2 token refreshed, expires in %ds", expires_in)
        except AtlassianAuthError:
            raise
        except Exception as e:
            raise AtlassianAuthError(f"Token refresh error: {e}") from e

    async def health_check(self) -> AuthHealthStatus:
        start = time.time()
        try:
            auth = await self.get_auth_headers()
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/wiki/rest/api/space?limit=1"
                async with session.get(url, headers=auth.headers) as resp:
                    elapsed = (time.time() - start) * 1000
                    if resp.status == 200:
                        return AuthHealthStatus(
                            is_healthy=True,
                            auth_method=AuthMethod.OAUTH2.value,
                            deployment_type="cloud",
                            response_time_ms=elapsed,
                        )
                    body = await resp.text()
                    return AuthHealthStatus(
                        is_healthy=False,
                        auth_method=AuthMethod.OAUTH2.value,
                        deployment_type="cloud",
                        response_time_ms=elapsed,
                        error_message=f"HTTP {resp.status}: {body[:200]}",
                    )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return AuthHealthStatus(
                is_healthy=False,
                auth_method=AuthMethod.OAUTH2.value,
                deployment_type="cloud",
                response_time_ms=elapsed,
                error_message=str(e),
            )
