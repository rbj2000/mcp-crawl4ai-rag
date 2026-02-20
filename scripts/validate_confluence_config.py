#!/usr/bin/env python3
"""Validate Confluence configuration from environment variables.

Usage:
    python scripts/validate_confluence_config.py
"""

import asyncio
import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.atlassian import ConfluenceConfig, AtlassianAuthFactory, AtlassianConfigError


async def main():
    print("Validating Confluence configuration...\n")

    # 1. Load config
    try:
        config = ConfluenceConfig.from_env()
    except AtlassianConfigError as e:
        print(f"FAIL  Configuration error: {e}")
        sys.exit(1)

    print(f"  URL:             {config.base_url}")
    print(f"  Deployment type: {config.deployment_type.value}")
    print(f"  Auth method:     {config.auth_method.value}")
    print(f"  SSL verify:      {config.ssl_verify}")
    print(f"  Pool size:       {config.pool_size}")
    print(f"  Max retries:     {config.max_retries}")
    print()

    # 2. Create auth provider
    try:
        auth = AtlassianAuthFactory.create_auth_provider(config)
    except Exception as e:
        print(f"FAIL  Could not create auth provider: {e}")
        sys.exit(1)

    print(f"  Auth provider:   {type(auth).__name__}")

    # 3. Health check (requires network)
    try:
        await auth.initialize()
        health = await auth.health_check()
        await auth.close()
        if health.is_healthy:
            print(f"  Health check:    PASS ({health.response_time_ms:.0f}ms)")
        else:
            print(f"  Health check:    FAIL - {health.error_message}")
    except Exception as e:
        print(f"  Health check:    SKIP (could not connect: {e})")

    print("\nConfiguration is valid.")


if __name__ == "__main__":
    asyncio.run(main())
