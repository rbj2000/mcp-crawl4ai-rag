"""Abstract base class for Atlassian authentication providers"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..base import AuthHeaders, AuthHealthStatus

logger = logging.getLogger(__name__)


class AtlassianAuthProvider(ABC):
    """Abstract base class for Atlassian authentication providers.

    Follows the pattern from ``src/ai_providers/base.py`` (EmbeddingProvider):
    ``__init__`` accepts a config dict and calls ``_validate_config()`` immediately.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration.

        Raises:
            AtlassianConfigError: If configuration is invalid.
        """

    async def initialize(self) -> None:
        """Initialize any async resources (default no-op)."""

    async def close(self) -> None:
        """Release any async resources (default no-op)."""

    @abstractmethod
    async def get_auth_headers(self) -> AuthHeaders:
        """Return current authentication headers.

        Returns:
            AuthHeaders with the appropriate Authorization header.
        """

    @abstractmethod
    async def health_check(self) -> AuthHealthStatus:
        """Verify credentials are valid by making a lightweight API call.

        Returns:
            AuthHealthStatus with result details.
        """

    @abstractmethod
    async def is_token_valid(self) -> bool:
        """Check whether the current token/credentials are still valid.

        Returns:
            True if valid, False otherwise.
        """

    async def refresh_if_needed(self) -> None:
        """Refresh credentials if they are about to expire.

        Default is a no-op. OAuth2 overrides this.
        """
