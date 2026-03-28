"""
OAuth Callback Handler for TradeStation PKCE Authentication

This module provides a simple HTTP server to handle OAuth 2.0 callbacks
from TradeStation's authorization server during the PKCE flow.

Usage:
    from src.execution.tradestation.auth.callback_handler import CallbackHandler
    from src.execution.tradestation.auth.oauth import OAuth2Client

    # Create OAuth client
    oauth_client = OAuth2Client(client_id="...", redirect_uri="http://localhost:8080")

    # Create callback handler
    handler = CallbackHandler(oauth_client)

    # Start server and wait for callback
    token_response = await handler.wait_for_callback()
    print(f"Got access token: {token_response.access_token}")
"""

import asyncio
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Callable, Optional

from src.execution.tradestation.auth.oauth import OAuth2Client
from src.execution.tradestation.exceptions import AuthError
from src.execution.tradestation.models import TokenResponse
from src.execution.tradestation.utils import setup_logger


class CallbackHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for OAuth callback.

    Receives the authorization code from TradeStation's OAuth callback
    and exchanges it for an access token.
    """

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger instead of stderr."""
        logger = setup_logger(f"{__name__}.CallbackHandler")
        logger.info(format % args)

    def do_GET(self):
        """Handle GET request from OAuth callback."""
        # Parse the URL
        parsed = urlparse(self.path)

        # Extract query parameters
        params = parse_qs(parsed.query)

        # Check for errors
        if "error" in params:
            error = params["error"][0]
            error_description = params.get("error_description", [""])[0]
            self.send_error_response(400, f"Authorization failed: {error} - {error_description}")
            return

        # Check for authorization code
        if "code" not in params:
            self.send_error_response(400, "Missing authorization code")
            return

        authorization_code = params["code"][0]

        # Validate state if present
        if "state" in params:
            # In production, validate state matches what we sent
            state = params["state"][0]
            # TODO: Implement state validation
            pass

        # Get the OAuth client from server instance
        oauth_client = getattr(self.server, "oauth_client", None)
        if not oauth_client:
            self.send_error_response(500, "OAuth client not configured")
            return

        # Exchange code for token
        try:
            # Run async operation in a new thread with new event loop
            import concurrent.futures
            import threading

            def run_async_in_thread():
                """Run async operation in a new thread with its own event loop."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        oauth_client.exchange_code_for_token(authorization_code)
                    )
                finally:
                    loop.close()

            # Execute in thread pool to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                token_response = future.result(timeout=30)

            # Store result for retrieval
            setattr(self.server, "token_response", token_response)

            # Send success response
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            # Send HTML response
            html = """
            <html>
            <head><title>Authentication Successful</title></head>
            <body>
                <h1>✅ Authentication Successful!</h1>
                <p>You can close this window and return to the application.</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

            # Signal server to stop
            setattr(self.server, "should_stop", True)

        except Exception as e:
            logger = setup_logger(f"{__name__}.CallbackHandler")
            logger.error(f"Error exchanging code for token: {e}")
            self.send_error_response(500, f"Error exchanging code: {e}")

    def send_error_response(self, code: int, message: str) -> None:
        """Send error response to client."""
        self.send_response(code)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        html = f"""
        <html>
        <head><title>Error {code}</title></head>
        <body>
            <h1>❌ Error {code}</h1>
            <p>{message}</p>
        </body>
        </html>
        """
        self.wfile.write(html.encode("utf-8"))


class OAuthCallbackServer:
    """
    Simple HTTP server to handle OAuth callback.

    This server listens on the specified port for the OAuth callback
    from TradeStation's authorization server.

    Attributes:
        oauth_client: OAuth2Client instance
        port: Port to listen on
        host: Host to bind to

    Example:
        oauth_client = OAuth2Client(client_id="...", redirect_uri="http://localhost:8080")
        server = OAuthCallbackServer(oauth_client, port=8080)

        # Start server in background
        import threading
        server_thread = threading.Thread(target=server.start)
        server_thread.start()

        # Wait for callback
        token_response = server.wait_for_callback()
        print(f"Got token: {token_response.access_token}")

        server.stop()
    """

    def __init__(
        self,
        oauth_client: OAuth2Client,
        port: int = 8080,
        host: str = "localhost",
    ) -> None:
        """
        Initialize OAuth callback server.

        Args:
            oauth_client: OAuth2Client instance for token exchange
            port: Port to listen on (default: 8080)
            host: Host to bind to (default: localhost)
        """
        self.oauth_client = oauth_client
        self.port = port
        self.host = host
        self.logger = setup_logger(f"{__name__}.OAuthCallbackServer")

        # Create server
        self.server = HTTPServer((host, port), CallbackHandler)
        self.server.oauth_client = oauth_client
        self.server.should_stop = False
        self.server.token_response = None

    def start(self, timeout: float = 300.0) -> TokenResponse | None:
        """
        Start the server and wait for OAuth callback.

        Args:
            timeout: Maximum time to wait for callback in seconds (default: 5 minutes)

        Returns:
            TokenResponse if callback received, None if timeout

        Example:
            token_response = server.start(timeout=120)
            if token_response:
                print(f"Got token: {token_response.access_token}")
        """
        import socket

        self.server.socket.settimeout(timeout)
        self.logger.info(f"OAuth callback server listening on {self.host}:{self.port}")
        self.logger.info(f"Waiting for callback (timeout: {timeout}s)...")

        try:
            self.server.serve_forever()
        except socket.timeout:
            self.logger.warning(f"Timeout waiting for OAuth callback after {timeout}s")
            return None
        except KeyboardInterrupt:
            self.logger.info("Server interrupted")
            return None
        finally:
            self.server.server_close()

        # Return token response if received
        return getattr(self.server, "token_response", None)

    def stop(self) -> None:
        """Stop the server gracefully."""
        self.logger.info("Stopping OAuth callback server")
        self.server.shutdown()
        self.server.server_close()

    def wait_for_callback(self, timeout: float = 300.0) -> TokenResponse | None:
        """
        Wait for OAuth callback (alias for start method).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            TokenResponse if successful, None otherwise
        """
        return self.start(timeout=timeout)
