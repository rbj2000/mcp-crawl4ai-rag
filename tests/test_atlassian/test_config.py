"""Tests for src/atlassian/config.py — ConfluenceConfig.from_env()"""

import os
import pytest

from src.atlassian.base import AuthMethod, DeploymentType, AtlassianConfigError
from src.atlassian.config import ConfluenceConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all CONFLUENCE_* env vars before each test."""
    for key in list(os.environ):
        if key.startswith("CONFLUENCE_"):
            monkeypatch.delenv(key, raising=False)


class TestDeploymentDetection:
    def test_cloud_auto_detect(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://myorg.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "user@example.com")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "tok")
        cfg = ConfluenceConfig.from_env()
        assert cfg.deployment_type == DeploymentType.CLOUD

    def test_on_prem_auto_detect(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://confluence.corp.local")
        monkeypatch.setenv("CONFLUENCE_PAT", "mypat")
        cfg = ConfluenceConfig.from_env()
        assert cfg.deployment_type == DeploymentType.ON_PREM

    def test_explicit_deployment_type(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://myorg.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_DEPLOYMENT_TYPE", "data_center")
        monkeypatch.setenv("CONFLUENCE_PAT", "tok")
        cfg = ConfluenceConfig.from_env()
        assert cfg.deployment_type == DeploymentType.DATA_CENTER

    def test_invalid_deployment_type(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_DEPLOYMENT_TYPE", "invalid")
        with pytest.raises(AtlassianConfigError, match="Invalid CONFLUENCE_DEPLOYMENT_TYPE"):
            ConfluenceConfig.from_env()


class TestAuthMethodDetection:
    def test_basic_auth(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "u")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "t")
        cfg = ConfluenceConfig.from_env()
        assert cfg.auth_method == AuthMethod.BASIC

    def test_pat_auth(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://confluence.corp.local")
        monkeypatch.setenv("CONFLUENCE_PAT", "mypat")
        cfg = ConfluenceConfig.from_env()
        assert cfg.auth_method == AuthMethod.PAT

    def test_oauth2_auth(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_CLIENT_ID", "cid")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_CLIENT_SECRET", "sec")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_ACCESS_TOKEN", "at")
        cfg = ConfluenceConfig.from_env()
        assert cfg.auth_method == AuthMethod.OAUTH2

    def test_explicit_auth_method(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_AUTH_METHOD", "pat")
        monkeypatch.setenv("CONFLUENCE_PAT", "mypat")
        cfg = ConfluenceConfig.from_env()
        assert cfg.auth_method == AuthMethod.PAT

    def test_no_credentials_raises(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        with pytest.raises(AtlassianConfigError, match="Cannot auto-detect"):
            ConfluenceConfig.from_env()


class TestValidation:
    def test_missing_url(self, monkeypatch):
        with pytest.raises(AtlassianConfigError, match="CONFLUENCE_URL is required"):
            ConfluenceConfig.from_env()

    def test_invalid_url_scheme(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "ftp://bad")
        with pytest.raises(AtlassianConfigError, match="must start with http"):
            ConfluenceConfig.from_env()

    def test_oauth2_on_prem_rejected(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://confluence.corp.local")
        monkeypatch.setenv("CONFLUENCE_AUTH_METHOD", "oauth2")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_CLIENT_ID", "cid")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_CLIENT_SECRET", "sec")
        monkeypatch.setenv("CONFLUENCE_OAUTH2_ACCESS_TOKEN", "at")
        with pytest.raises(AtlassianConfigError, match="OAuth2 is only supported"):
            ConfluenceConfig.from_env()

    def test_basic_missing_username(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_AUTH_METHOD", "basic")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "tok")
        with pytest.raises(AtlassianConfigError, match="CONFLUENCE_USERNAME"):
            ConfluenceConfig.from_env()

    def test_ca_bundle_not_found(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "u")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "t")
        monkeypatch.setenv("CONFLUENCE_CA_BUNDLE", "/nonexistent/cert.pem")
        with pytest.raises(AtlassianConfigError, match="CA bundle file not found"):
            ConfluenceConfig.from_env()

    def test_url_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("CONFLUENCE_URL", "https://x.atlassian.net/")
        monkeypatch.setenv("CONFLUENCE_USERNAME", "u")
        monkeypatch.setenv("CONFLUENCE_API_TOKEN", "t")
        cfg = ConfluenceConfig.from_env()
        assert not cfg.base_url.endswith("/")
