"""Base types, enums, dataclasses, and exceptions for Atlassian integration"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """Atlassian deployment types"""
    CLOUD = "cloud"
    ON_PREM = "on_prem"
    DATA_CENTER = "data_center"


class AuthMethod(Enum):
    """Supported authentication methods"""
    BASIC = "basic"
    PAT = "pat"
    OAUTH2 = "oauth2"


@dataclass
class AuthHeaders:
    """Authentication headers for Atlassian API requests"""
    headers: Dict[str, str]
    auth_method: AuthMethod
    expires_at: Optional[float] = None


@dataclass
class AuthHealthStatus:
    """Health check status for Atlassian authentication"""
    is_healthy: bool
    auth_method: str
    deployment_type: str
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = field(default_factory=dict)


# --- Exceptions ---

class AtlassianError(Exception):
    """Base exception for all Atlassian-related errors"""
    pass


class AtlassianAuthError(AtlassianError):
    """Authentication failure (invalid credentials, expired token, etc.)"""
    pass


class AtlassianConfigError(AtlassianError):
    """Configuration validation error"""
    pass


class AtlassianAPIError(AtlassianError):
    """Atlassian REST API error"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AtlassianRateLimitError(AtlassianAPIError):
    """Rate limit exceeded (HTTP 429)"""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
