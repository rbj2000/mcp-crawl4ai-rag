"""Configuration management for Atlassian / Confluence integration"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .base import AuthMethod, DeploymentType, AtlassianConfigError

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceConfig:
    """Confluence connection configuration.

    Use ``ConfluenceConfig.from_env()`` to create an instance from environment
    variables. Explicit env vars always override auto-detection.
    """

    base_url: str
    deployment_type: DeploymentType
    auth_method: AuthMethod
    auth_config: Dict[str, Any] = field(default_factory=dict)

    # SSL
    ssl_verify: bool = True
    ca_bundle: Optional[str] = None

    # Timeouts
    timeout: float = 30.0
    connect_timeout: float = 10.0
    pool_size: int = 10

    # Rate-limit / retry
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "ConfluenceConfig":
        """Build configuration from ``CONFLUENCE_*`` environment variables.

        Raises:
            AtlassianConfigError: on invalid / missing configuration.
        """
        base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
        if not base_url:
            raise AtlassianConfigError("CONFLUENCE_URL is required")
        if not base_url.startswith(("http://", "https://")):
            raise AtlassianConfigError(
                "CONFLUENCE_URL must start with http:// or https://"
            )

        deployment_type = cls._resolve_deployment_type(base_url)
        auth_method = cls._resolve_auth_method(deployment_type)
        auth_config = cls._build_auth_config(auth_method, base_url)

        # SSL
        ssl_verify = os.getenv("CONFLUENCE_SSL_VERIFY", "true").lower() == "true"
        ca_bundle = os.getenv("CONFLUENCE_CA_BUNDLE")
        if ca_bundle and not os.path.isfile(ca_bundle):
            raise AtlassianConfigError(
                f"CA bundle file not found: {ca_bundle}"
            )

        # Timeouts / pool
        timeout = float(os.getenv("CONFLUENCE_TIMEOUT", "30.0"))
        connect_timeout = float(os.getenv("CONFLUENCE_CONNECT_TIMEOUT", "10.0"))
        pool_size = int(os.getenv("CONFLUENCE_POOL_SIZE", "10"))
        max_retries = int(os.getenv("CONFLUENCE_MAX_RETRIES", "3"))

        config = cls(
            base_url=base_url,
            deployment_type=deployment_type,
            auth_method=auth_method,
            auth_config=auth_config,
            ssl_verify=ssl_verify,
            ca_bundle=ca_bundle,
            timeout=timeout,
            connect_timeout=connect_timeout,
            pool_size=pool_size,
            max_retries=max_retries,
        )
        config._validate()
        return config

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_deployment_type(url: str) -> DeploymentType:
        explicit = os.getenv("CONFLUENCE_DEPLOYMENT_TYPE", "").lower()
        if explicit:
            try:
                return DeploymentType(explicit)
            except ValueError:
                valid = [d.value for d in DeploymentType]
                raise AtlassianConfigError(
                    f"Invalid CONFLUENCE_DEPLOYMENT_TYPE='{explicit}'. Valid: {valid}"
                )
        # Auto-detect
        if ".atlassian.net" in url:
            return DeploymentType.CLOUD
        return DeploymentType.ON_PREM

    @staticmethod
    def _resolve_auth_method(deployment_type: DeploymentType) -> AuthMethod:
        explicit = os.getenv("CONFLUENCE_AUTH_METHOD", "").lower()
        if explicit:
            try:
                return AuthMethod(explicit)
            except ValueError:
                valid = [a.value for a in AuthMethod]
                raise AtlassianConfigError(
                    f"Invalid CONFLUENCE_AUTH_METHOD='{explicit}'. Valid: {valid}"
                )

        # Auto-detect: OAuth2 > PAT > Basic
        if os.getenv("CONFLUENCE_OAUTH2_CLIENT_ID"):
            method = AuthMethod.OAUTH2
        elif os.getenv("CONFLUENCE_PAT") or os.getenv("CONFLUENCE_API_TOKEN"):
            if deployment_type == DeploymentType.CLOUD:
                # On Cloud with only a token, assume Basic (email+token)
                if os.getenv("CONFLUENCE_USERNAME"):
                    method = AuthMethod.BASIC
                else:
                    method = AuthMethod.PAT
            else:
                method = AuthMethod.PAT
        elif os.getenv("CONFLUENCE_USERNAME"):
            method = AuthMethod.BASIC
        else:
            raise AtlassianConfigError(
                "Cannot auto-detect auth method. Set CONFLUENCE_AUTH_METHOD or "
                "provide credentials via CONFLUENCE_USERNAME/CONFLUENCE_API_TOKEN, "
                "CONFLUENCE_PAT, or CONFLUENCE_OAUTH2_CLIENT_ID."
            )
        return method

    @staticmethod
    def _build_auth_config(auth_method: AuthMethod, base_url: str) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"base_url": base_url}

        if auth_method == AuthMethod.BASIC:
            cfg["username"] = os.getenv("CONFLUENCE_USERNAME", "")
            cfg["api_token"] = os.getenv("CONFLUENCE_API_TOKEN", "")

        elif auth_method == AuthMethod.PAT:
            cfg["pat"] = os.getenv("CONFLUENCE_PAT") or os.getenv("CONFLUENCE_API_TOKEN", "")

        elif auth_method == AuthMethod.OAUTH2:
            cfg["oauth2_client_id"] = os.getenv("CONFLUENCE_OAUTH2_CLIENT_ID", "")
            cfg["oauth2_client_secret"] = os.getenv("CONFLUENCE_OAUTH2_CLIENT_SECRET", "")
            cfg["oauth2_access_token"] = os.getenv("CONFLUENCE_OAUTH2_ACCESS_TOKEN")
            cfg["oauth2_refresh_token"] = os.getenv("CONFLUENCE_OAUTH2_REFRESH_TOKEN")
            cfg["oauth2_token_expires_at"] = os.getenv("CONFLUENCE_OAUTH2_TOKEN_EXPIRES_AT")
            cfg["cloud_id"] = os.getenv("CONFLUENCE_CLOUD_ID")

        return cfg

    # ------------------------------------------------------------------
    # Cross-field validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.auth_method == AuthMethod.OAUTH2 and self.deployment_type != DeploymentType.CLOUD:
            raise AtlassianConfigError(
                "OAuth2 is only supported for Atlassian Cloud deployments"
            )

        if self.auth_method == AuthMethod.BASIC:
            if not self.auth_config.get("username"):
                raise AtlassianConfigError("CONFLUENCE_USERNAME is required for Basic auth")
            if not self.auth_config.get("api_token"):
                raise AtlassianConfigError("CONFLUENCE_API_TOKEN is required for Basic auth")

        elif self.auth_method == AuthMethod.PAT:
            if not self.auth_config.get("pat"):
                raise AtlassianConfigError(
                    "CONFLUENCE_PAT or CONFLUENCE_API_TOKEN is required for PAT auth"
                )

        elif self.auth_method == AuthMethod.OAUTH2:
            if not self.auth_config.get("oauth2_client_id"):
                raise AtlassianConfigError("CONFLUENCE_OAUTH2_CLIENT_ID is required")
            if not self.auth_config.get("oauth2_client_secret"):
                raise AtlassianConfigError("CONFLUENCE_OAUTH2_CLIENT_SECRET is required")
            if not self.auth_config.get("oauth2_access_token") and not self.auth_config.get("oauth2_refresh_token"):
                raise AtlassianConfigError(
                    "At least CONFLUENCE_OAUTH2_ACCESS_TOKEN or "
                    "CONFLUENCE_OAUTH2_REFRESH_TOKEN is required"
                )

        if self.timeout <= 0:
            raise AtlassianConfigError("CONFLUENCE_TIMEOUT must be positive")
        if self.connect_timeout <= 0:
            raise AtlassianConfigError("CONFLUENCE_CONNECT_TIMEOUT must be positive")
        if self.pool_size <= 0:
            raise AtlassianConfigError("CONFLUENCE_POOL_SIZE must be positive")
