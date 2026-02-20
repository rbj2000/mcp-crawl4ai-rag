"""Atlassian authentication and HTTP client package"""

from .base import (
    AuthHeaders,
    AuthHealthStatus,
    AuthMethod,
    AtlassianAPIError,
    AtlassianAuthError,
    AtlassianConfigError,
    AtlassianError,
    AtlassianRateLimitError,
    DeploymentType,
)
from .auth import (
    AtlassianAuthProvider,
    BasicAuthProvider,
    OAuth2AuthProvider,
    PATAuthProvider,
)
from .config import ConfluenceConfig
from .factory import AtlassianAuthFactory
from .http_client import AtlassianHTTPClient

__all__ = [
    # Enums
    "DeploymentType",
    "AuthMethod",
    # Dataclasses
    "AuthHeaders",
    "AuthHealthStatus",
    # Exceptions
    "AtlassianError",
    "AtlassianAuthError",
    "AtlassianConfigError",
    "AtlassianAPIError",
    "AtlassianRateLimitError",
    # Auth providers
    "AtlassianAuthProvider",
    "BasicAuthProvider",
    "PATAuthProvider",
    "OAuth2AuthProvider",
    # Config
    "ConfluenceConfig",
    # Factory & client
    "AtlassianAuthFactory",
    "AtlassianHTTPClient",
]
