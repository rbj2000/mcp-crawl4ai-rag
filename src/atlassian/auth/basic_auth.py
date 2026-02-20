"""Basic authentication provider for Atlassian Cloud (email + API token)"""

import base64
import logging
import time
from typing import Dict, Any

import aiohttp

from ..base import AuthHeaders, AuthHealthStatus, AuthMethod, AtlassianConfigError
from .base_auth import AtlassianAuthProvider

logger = logging.getLogger(__name__)


class BasicAuthProvider(AtlassianAuthProvider):
    """Basic Auth for Atlassian Cloud: ``Authorization: Basic base64(email:api_token)``"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.username = config["username"]
        self.api_token = config["api_token"]
        self.base_url = config["base_url"]

    def _validate_config(self) -> None:
        missing = []
        if not self.config.get("username"):
            missing.append("username")
        if not self.config.get("api_token"):
            missing.append("api_token")
        if not self.config.get("base_url"):
            missing.append("base_url")
        if missing:
            raise AtlassianConfigError(
                f"BasicAuthProvider missing required config: {missing}"
            )

    async def get_auth_headers(self) -> AuthHeaders:
        credentials = base64.b64encode(
            f"{self.username}:{self.api_token}".encode()
        ).decode()
        return AuthHeaders(
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            auth_method=AuthMethod.BASIC,
        )

    async def is_token_valid(self) -> bool:
        # API tokens do not expire
        return True

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
                            auth_method=AuthMethod.BASIC.value,
                            deployment_type="cloud",
                            response_time_ms=elapsed,
                        )
                    body = await resp.text()
                    return AuthHealthStatus(
                        is_healthy=False,
                        auth_method=AuthMethod.BASIC.value,
                        deployment_type="cloud",
                        response_time_ms=elapsed,
                        error_message=f"HTTP {resp.status}: {body[:200]}",
                    )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return AuthHealthStatus(
                is_healthy=False,
                auth_method=AuthMethod.BASIC.value,
                deployment_type="cloud",
                response_time_ms=elapsed,
                error_message=str(e),
            )
