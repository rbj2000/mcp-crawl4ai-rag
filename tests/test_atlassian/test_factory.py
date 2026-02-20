"""Tests for AtlassianAuthFactory"""

import pytest

from src.atlassian.base import AuthMethod, AtlassianConfigError
from src.atlassian.auth.basic_auth import BasicAuthProvider
from src.atlassian.auth.pat_auth import PATAuthProvider
from src.atlassian.auth.oauth2_auth import OAuth2AuthProvider
from src.atlassian.config import ConfluenceConfig
from src.atlassian.factory import AtlassianAuthFactory


class TestRegistration:
    def test_built_in_providers_registered(self):
        registered = AtlassianAuthFactory.list_registered()
        assert "basic" in registered
        assert "pat" in registered
        assert "oauth2" in registered

    def test_create_basic_provider(self):
        from src.atlassian.base import DeploymentType
        config = ConfluenceConfig(
            base_url="https://x.atlassian.net",
            deployment_type=DeploymentType.CLOUD,
            auth_method=AuthMethod.BASIC,
            auth_config={
                "username": "user@example.com",
                "api_token": "tok",
                "base_url": "https://x.atlassian.net",
            },
        )
        provider = AtlassianAuthFactory.create_auth_provider(config)
        assert isinstance(provider, BasicAuthProvider)

    def test_create_pat_provider(self):
        from src.atlassian.base import DeploymentType
        config = ConfluenceConfig(
            base_url="https://confluence.corp.local",
            deployment_type=DeploymentType.ON_PREM,
            auth_method=AuthMethod.PAT,
            auth_config={
                "pat": "my-pat",
                "base_url": "https://confluence.corp.local",
            },
        )
        provider = AtlassianAuthFactory.create_auth_provider(config)
        assert isinstance(provider, PATAuthProvider)

    def test_create_http_client(self):
        from src.atlassian.base import DeploymentType
        from src.atlassian.http_client import AtlassianHTTPClient
        config = ConfluenceConfig(
            base_url="https://confluence.corp.local",
            deployment_type=DeploymentType.ON_PREM,
            auth_method=AuthMethod.PAT,
            auth_config={
                "pat": "my-pat",
                "base_url": "https://confluence.corp.local",
            },
        )
        client = AtlassianAuthFactory.create_http_client(config)
        assert isinstance(client, AtlassianHTTPClient)


class TestClearRegistry:
    def test_clear_and_re_register(self):
        AtlassianAuthFactory.clear_registry()
        assert AtlassianAuthFactory.list_registered() == []

        # Re-register so other tests aren't affected
        AtlassianAuthFactory.register_auth_provider(AuthMethod.BASIC, BasicAuthProvider)
        AtlassianAuthFactory.register_auth_provider(AuthMethod.PAT, PATAuthProvider)
        AtlassianAuthFactory.register_auth_provider(AuthMethod.OAUTH2, OAuth2AuthProvider)
        assert len(AtlassianAuthFactory.list_registered()) == 3
