"""HTTP client for Atlassian APIs with retry, rate-limit, and auth injection"""

import asyncio
import logging
import ssl
import time
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from .auth.base_auth import AtlassianAuthProvider
from .base import (
    AtlassianAPIError,
    AtlassianAuthError,
    AtlassianRateLimitError,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class AtlassianHTTPClient:
    """Async HTTP client that injects auth headers and handles retries.

    Follows aiohttp patterns from ``src/ai_providers/providers/ollama_provider.py``.
    """

    def __init__(
        self,
        auth_provider: AtlassianAuthProvider,
        base_url: str,
        *,
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        pool_size: int = 10,
        max_retries: int = 3,
        ssl_verify: bool = True,
        ca_bundle: Optional[str] = None,
    ):
        self.auth_provider = auth_provider
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.pool_size = pool_size
        self.max_retries = max_retries
        self.ssl_verify = ssl_verify
        self.ca_bundle = ca_bundle

        self._session: Optional[ClientSession] = None
        self._ssl_context: Optional[ssl.SSLContext] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the ``aiohttp.ClientSession`` and SSL context."""
        self._ssl_context = self._build_ssl_context()
        connector = TCPConnector(
            limit=self.pool_size,
            ssl=self._ssl_context,
        )
        client_timeout = ClientTimeout(
            total=self.timeout,
            connect=self.connect_timeout,
        )
        self._session = ClientSession(
            timeout=client_timeout,
            connector=connector,
        )
        await self.auth_provider.initialize()
        logger.info("AtlassianHTTPClient initialized (base_url=%s)", self.base_url)

    async def close(self) -> None:
        """Close session and auth provider."""
        if self._session:
            await self._session.close()
            self._session = None
        await self.auth_provider.close()
        logger.info("AtlassianHTTPClient closed")

    # ------------------------------------------------------------------
    # Public request API
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Send an authenticated request with retry logic.

        Args:
            method: HTTP method (GET, POST, …).
            path: API path appended to ``base_url``.
            params: Query parameters.
            json: JSON body.
            headers: Extra headers (merged with auth headers).

        Returns:
            Parsed JSON response body.

        Raises:
            AtlassianAuthError: On 401 after one retry with refreshed token.
            AtlassianRateLimitError: On 429 after retries exhausted.
            AtlassianAPIError: On other non-recoverable HTTP errors.
        """
        if not self._session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        url = f"{self.base_url}{path}"
        retry_delay = 1.0
        auth_retried = False

        for attempt in range(self.max_retries + 1):
            auth = await self.auth_provider.get_auth_headers()
            merged_headers = {**auth.headers, **(headers or {})}

            self._log_request(method, url, attempt)

            try:
                async with self._session.request(
                    method, url, params=params, json=json, headers=merged_headers
                ) as resp:
                    status = resp.status

                    if 200 <= status < 300:
                        if resp.content_type and "json" in resp.content_type:
                            return await resp.json()
                        text = await resp.text()
                        return {"_raw": text}

                    body = await resp.text()

                    # 401 — force refresh once then retry
                    if status == 401 and not auth_retried:
                        auth_retried = True
                        logger.warning("Got 401, forcing token refresh and retrying")
                        await self.auth_provider.refresh_if_needed()
                        continue

                    if status == 401:
                        raise AtlassianAuthError(
                            f"Authentication failed after refresh (HTTP 401): {body[:300]}"
                        )

                    # 429 — respect Retry-After
                    if status == 429:
                        retry_after = self._parse_retry_after(resp)
                        if attempt < self.max_retries:
                            wait = retry_after or retry_delay
                            logger.warning(
                                "Rate limited (429). Waiting %.1fs (attempt %d/%d)",
                                wait, attempt + 1, self.max_retries + 1,
                            )
                            await asyncio.sleep(wait)
                            retry_delay = min(retry_delay * 2, 60.0)
                            continue
                        raise AtlassianRateLimitError(
                            f"Rate limit exceeded after {self.max_retries + 1} attempts",
                            retry_after=retry_after,
                            status_code=429,
                            response_body=body[:500],
                        )

                    # Retryable server errors
                    if status in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                        logger.warning(
                            "Retryable error %d on %s (attempt %d/%d). Retrying in %.1fs",
                            status, path, attempt + 1, self.max_retries + 1, retry_delay,
                        )
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 60.0)
                        continue

                    raise AtlassianAPIError(
                        f"HTTP {status}: {body[:500]}",
                        status_code=status,
                        response_body=body[:1000],
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Connection error on %s (attempt %d/%d): %s. Retrying in %.1fs",
                        path, attempt + 1, self.max_retries + 1, e, retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    continue
                raise AtlassianAPIError(f"Connection failed after retries: {e}") from e

        # Should not reach here, but just in case
        raise AtlassianAPIError("Request failed after all retries")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ssl_context(self) -> Optional[ssl.SSLContext]:
        if not self.ssl_verify:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        if self.ca_bundle:
            ctx = ssl.create_default_context(cafile=self.ca_bundle)
            return ctx
        return None  # aiohttp uses default system certs

    @staticmethod
    def _parse_retry_after(resp: aiohttp.ClientResponse) -> Optional[float]:
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _log_request(method: str, url: str, attempt: int) -> None:
        logger.debug("→ %s %s (attempt %d)", method, url, attempt + 1)
