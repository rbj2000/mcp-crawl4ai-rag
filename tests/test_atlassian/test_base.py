"""Tests for src/atlassian/base.py — enums, dataclasses, exceptions"""

import pytest

from src.atlassian.base import (
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


class TestEnums:
    def test_deployment_type_values(self):
        assert DeploymentType.CLOUD.value == "cloud"
        assert DeploymentType.ON_PREM.value == "on_prem"
        assert DeploymentType.DATA_CENTER.value == "data_center"

    def test_auth_method_values(self):
        assert AuthMethod.BASIC.value == "basic"
        assert AuthMethod.PAT.value == "pat"
        assert AuthMethod.OAUTH2.value == "oauth2"


class TestAuthHeaders:
    def test_creation(self):
        ah = AuthHeaders(
            headers={"Authorization": "Basic abc"},
            auth_method=AuthMethod.BASIC,
        )
        assert ah.headers["Authorization"] == "Basic abc"
        assert ah.auth_method == AuthMethod.BASIC
        assert ah.expires_at is None

    def test_with_expiry(self):
        ah = AuthHeaders(
            headers={}, auth_method=AuthMethod.OAUTH2, expires_at=99999.0
        )
        assert ah.expires_at == 99999.0


class TestAuthHealthStatus:
    def test_healthy(self):
        s = AuthHealthStatus(
            is_healthy=True,
            auth_method="basic",
            deployment_type="cloud",
            response_time_ms=42.0,
        )
        assert s.is_healthy
        assert s.error_message is None

    def test_unhealthy(self):
        s = AuthHealthStatus(
            is_healthy=False,
            auth_method="pat",
            deployment_type="on_prem",
            response_time_ms=100.0,
            error_message="timeout",
        )
        assert not s.is_healthy
        assert s.error_message == "timeout"


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(AtlassianAuthError, AtlassianError)
        assert issubclass(AtlassianConfigError, AtlassianError)
        assert issubclass(AtlassianAPIError, AtlassianError)
        assert issubclass(AtlassianRateLimitError, AtlassianAPIError)

    def test_api_error_attrs(self):
        e = AtlassianAPIError("fail", status_code=500, response_body="err")
        assert e.status_code == 500
        assert e.response_body == "err"

    def test_rate_limit_attrs(self):
        e = AtlassianRateLimitError("slow", retry_after=30.0, status_code=429)
        assert e.retry_after == 30.0
        assert e.status_code == 429
