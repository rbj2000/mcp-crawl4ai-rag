"""Personal Access Token (PAT) authentication for Atlassian On-Prem / Data Center"""

import logging
import time
from typing import Dict, Any

import aiohttp

from ..base import AuthHeaders, AuthHealthStatus, AuthMethod, AtlassianConfigError
from .base_auth import AtlassianAuthProvider

logger = logging.getLogger(__name__)


class PATAuthProvider(AtlassianAuthProvider):
    """PAT Auth for On-Prem/DC: ``Authorization: Bearer {pat}``"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pat = config["pat"]
        self.base_url = config["base_url"]

    def _validate_config(self) -> None:
        if not self.config.get("pat"):
            raise AtlassianConfigError(
                "PATAuthProvider missing required config: pat"
            )
        if not self.config.get("base_url"):
            raise AtlassianConfigError(
                "PATAuthProvider missing required config: base_url"
            )

    async def get_auth_headers(self) -> AuthHeaders:
        return AuthHeaders(
            headers={
                "Authorization": f"Bearer {self.pat}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            auth_method=AuthMethod.PAT,
        )

    async def is_token_valid(self) -> bool:
        # PATs do not self-expire (admin can revoke them server-side)
        return True

    async def health_check(self) -> AuthHealthStatus:
        start = time.time()
        try:
            auth = await self.get_auth_headers()
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/rest/api/space?limit=1"
                async with session.get(url, headers=auth.headers) as resp:
                    elapsed = (time.time() - start) * 1000
                    if resp.status == 200:
                        return AuthHealthStatus(
                            is_healthy=True,
                            auth_method=AuthMethod.PAT.value,
                            deployment_type="on_prem",
                            response_time_ms=elapsed,
                        )
                    body = await resp.text()
                    return AuthHealthStatus(
                        is_healthy=False,
                        auth_method=AuthMethod.PAT.value,
                        deployment_type="on_prem",
                        response_time_ms=elapsed,
                        error_message=f"HTTP {resp.status}: {body[:200]}",
                    )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return AuthHealthStatus(
                is_healthy=False,
                auth_method=AuthMethod.PAT.value,
                deployment_type="on_prem",
                response_time_ms=elapsed,
                error_message=str(e),
            )
