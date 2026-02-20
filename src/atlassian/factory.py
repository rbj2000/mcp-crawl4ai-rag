"""Factory for creating Atlassian auth providers and HTTP clients"""

import logging
from typing import Dict, Type

from .auth.base_auth import AtlassianAuthProvider
from .auth.basic_auth import BasicAuthProvider
from .auth.pat_auth import PATAuthProvider
from .auth.oauth2_auth import OAuth2AuthProvider
from .base import AuthMethod, AtlassianConfigError
from .config import ConfluenceConfig
from .http_client import AtlassianHTTPClient

logger = logging.getLogger(__name__)


class AtlassianAuthFactory:
    """Registry-based factory for auth providers and convenience HTTP client creation.

    Follows the pattern from ``src/ai_providers/factory.py``.
    """

    _auth_providers: Dict[AuthMethod, Type[AtlassianAuthProvider]] = {}

    @classmethod
    def register_auth_provider(
        cls,
        method: AuthMethod,
        provider_class: Type[AtlassianAuthProvider],
    ) -> None:
        if not issubclass(provider_class, AtlassianAuthProvider):
            raise ValueError("provider_class must be a subclass of AtlassianAuthProvider")
        cls._auth_providers[method] = provider_class
        logger.info("Registered Atlassian auth provider: %s", method.value)

    @classmethod
    def create_auth_provider(cls, config: ConfluenceConfig) -> AtlassianAuthProvider:
        """Create the appropriate auth provider from a ``ConfluenceConfig``."""
        provider_cls = cls._auth_providers.get(config.auth_method)
        if provider_cls is None:
            available = [m.value for m in cls._auth_providers]
            raise AtlassianConfigError(
                f"No auth provider registered for method '{config.auth_method.value}'. "
                f"Available: {available}"
            )
        return provider_cls(config.auth_config)

    @classmethod
    def create_http_client(cls, config: ConfluenceConfig) -> AtlassianHTTPClient:
        """Convenience: config → auth provider → HTTP client."""
        auth = cls.create_auth_provider(config)
        return AtlassianHTTPClient(
            auth_provider=auth,
            base_url=config.base_url,
            timeout=config.timeout,
            connect_timeout=config.connect_timeout,
            pool_size=config.pool_size,
            max_retries=config.max_retries,
            ssl_verify=config.ssl_verify,
            ca_bundle=config.ca_bundle,
        )

    @classmethod
    def list_registered(cls) -> list:
        return [m.value for m in cls._auth_providers]

    @classmethod
    def clear_registry(cls) -> None:
        cls._auth_providers.clear()


def _register_providers() -> None:
    """Register built-in auth providers at module import time."""
    AtlassianAuthFactory.register_auth_provider(AuthMethod.BASIC, BasicAuthProvider)
    AtlassianAuthFactory.register_auth_provider(AuthMethod.PAT, PATAuthProvider)
    AtlassianAuthFactory.register_auth_provider(AuthMethod.OAUTH2, OAuth2AuthProvider)
    logger.info(
        "Atlassian auth provider registration complete. Available: %s",
        AtlassianAuthFactory.list_registered(),
    )


_register_providers()
