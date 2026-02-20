"""Atlassian authentication, HTTP client, and Confluence crawler package"""

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
from .confluence_crawler import ConfluenceCrawler, ConfluencePage, CrawlResult, CrawlSummary
from .content_converter import ADFConverter, XHTMLConverter, convert_content
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
    # Content converters
    "ADFConverter",
    "XHTMLConverter",
    "convert_content",
    # Confluence crawler
    "ConfluenceCrawler",
    "ConfluencePage",
    "CrawlResult",
    "CrawlSummary",
    # Factory & client
    "AtlassianAuthFactory",
    "AtlassianHTTPClient",
]
