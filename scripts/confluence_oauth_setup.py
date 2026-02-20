#!/usr/bin/env python3
"""One-time CLI tool for Atlassian OAuth 2.0 3LO browser flow.

Usage:
    python scripts/confluence_oauth_setup.py \
        --client-id YOUR_CLIENT_ID \
        --client-secret YOUR_CLIENT_SECRET \
        [--port 8089]

The script starts a local HTTP server, opens the Atlassian authorization URL
in the default browser, waits for the redirect callback, exchanges the
authorization code for tokens, and prints the resulting environment variables.
"""

import argparse
import http.server
import json
import os
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser

AUTHORIZE_URL = "https://auth.atlassian.com/authorize"
TOKEN_URL = "https://auth.atlassian.com/oauth/token"
SCOPES = [
    "read:confluence-content.all",
    "read:confluence-space.summary",
    "offline_access",  # needed for refresh_token
]


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Receives the OAuth redirect and extracts the authorization code."""

    authorization_code = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        if code:
            CallbackHandler.authorization_code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h2>Authorization successful!</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
            )
        else:
            error = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<h2>Error: {error}</h2>".encode())

    def log_message(self, format, *args):
        pass  # suppress logs


def exchange_code(client_id: str, client_secret: str, code: str, redirect_uri: str) -> dict:
    payload = json.dumps({
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "redirect_uri": redirect_uri,
    }).encode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Atlassian OAuth 2.0 setup")
    parser.add_argument("--client-id", required=True, help="OAuth2 client ID")
    parser.add_argument("--client-secret", required=True, help="OAuth2 client secret")
    parser.add_argument("--port", type=int, default=8089, help="Local callback port")
    args = parser.parse_args()

    redirect_uri = f"http://localhost:{args.port}/callback"

    # Build authorize URL
    params = urllib.parse.urlencode({
        "audience": "api.atlassian.com",
        "client_id": args.client_id,
        "scope": " ".join(SCOPES),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "prompt": "consent",
    })
    auth_url = f"{AUTHORIZE_URL}?{params}"

    # Start local server
    server = http.server.HTTPServer(("localhost", args.port), CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print(f"Opening browser for authorization...\n{auth_url}\n")
    webbrowser.open(auth_url)
    print("Waiting for callback...")
    thread.join(timeout=120)
    server.server_close()

    code = CallbackHandler.authorization_code
    if not code:
        print("ERROR: No authorization code received.", file=sys.stderr)
        sys.exit(1)

    print("Exchanging authorization code for tokens...")
    tokens = exchange_code(args.client_id, args.client_secret, code, redirect_uri)

    print("\n# Add these to your .env file:")
    print(f"CONFLUENCE_OAUTH2_CLIENT_ID={args.client_id}")
    print(f"CONFLUENCE_OAUTH2_CLIENT_SECRET={args.client_secret}")
    print(f"CONFLUENCE_OAUTH2_ACCESS_TOKEN={tokens['access_token']}")
    if "refresh_token" in tokens:
        print(f"CONFLUENCE_OAUTH2_REFRESH_TOKEN={tokens['refresh_token']}")
    if "expires_in" in tokens:
        import time
        expires_at = int(time.time() + tokens["expires_in"])
        print(f"CONFLUENCE_OAUTH2_TOKEN_EXPIRES_AT={expires_at}")
    print()


if __name__ == "__main__":
    main()
