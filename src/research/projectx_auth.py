"""ProjectX API authentication for TopstepX execution.

Credentials are loaded from .projectx_api_key (two non-comment lines):
    line 1: TopstepX username (email address)
    line 2: ProjectX API key (generated in TopstepX Settings → API tab)

The API key is exchanged for a 24-hour JWT via POST to loginKey. The token
is cached and re-issued automatically 30 minutes before expiry.

Interface mirrors TradeStationAuthV3 (authenticate / start_auto_refresh /
is_authenticated / cleanup / from_file) so ProjectXClient can drop in
wherever TradeStationAuthV3 is accepted.

Exception divergence: ProjectXAuth raises ProjectXAuthError (a RuntimeError
subclass) rather than AuthenticationError from src.data.exceptions, to keep
this module self-contained. Call sites should catch Exception (or
ProjectXAuthError explicitly) rather than AuthenticationError.

Refresh interval: defaults to 60 min (vs TradeStationAuthV3's 10 min)
because ProjectX tokens are valid for 24 hours, not ~30 minutes.
"""

import asyncio
import base64
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_REFRESH_BUFFER = timedelta(minutes=30)
_RETRY_DELAYS = (1, 2, 4)  # seconds between retry attempts, exponential back-off


class ProjectXAuthError(RuntimeError):
    """Raised when ProjectX authentication fails and cannot be retried."""


class ProjectXAuth:
    """ProjectX API key → JWT authentication for TopstepX.

    Credentials file (.projectx_api_key) format::

        # lines starting with '#' are ignored
        your@email.com
        your-projectx-api-key

    Usage::

        auth = ProjectXAuth.from_file(".projectx_api_key")
        token = await auth.authenticate()          # bearer JWT
        await auth.start_auto_refresh()
        # ... trading loop ...
        await auth.cleanup()
    """

    AUTH_ENDPOINT = "https://api.topstepx.com/api/Auth/loginKey"
    VALIDATE_ENDPOINT = "https://api.topstepx.com/api/Auth/validate"

    def __init__(self, username: str, api_key: str) -> None:
        self._username = username
        self._api_key = api_key
        self._token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._refresh_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._should_stop = False
        self._closed = False
        self._login_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, key_file: str = ".projectx_api_key") -> "ProjectXAuth":
        """Load credentials from a two-line flat file.

        Args:
            key_file: Path to credentials file (default: ``.projectx_api_key``).

        Returns:
            ProjectXAuth instance ready for ``await auth.authenticate()``.

        Raises:
            ProjectXAuthError: If the file is missing or malformed.
        """
        path = Path(key_file)
        if not path.exists():
            raise ProjectXAuthError(
                f"Credentials file not found: {key_file}. "
                "Create it with username (email) on line 1 and API key on line 2."
            )
        try:
            lines = [
                ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        except Exception as exc:
            raise ProjectXAuthError(f"Cannot read {key_file}: {exc}") from exc

        if len(lines) < 2:
            raise ProjectXAuthError(
                f"{key_file} must have username on line 1 and API key on line 2 "
                "(blank lines and lines starting with '#' are ignored)."
            )
        if len(lines) > 2:
            logger.warning(
                f"{key_file} has {len(lines)} non-comment lines; only the first two are used."
            )
        return cls(username=lines[0], api_key=lines[1])

    # ------------------------------------------------------------------
    # Public interface (mirrors TradeStationAuthV3)
    # ------------------------------------------------------------------

    async def authenticate(self) -> str:
        """Return a valid bearer JWT, logging in if the cached token is expired.

        Concurrent callers are serialized via a lock so only one login request
        is in-flight at a time.

        Returns:
            JWT string for ``Authorization: Bearer <token>`` headers.

        Raises:
            ProjectXAuthError: If login fails after all retries.
        """
        if self._is_token_valid():
            return self._token  # type: ignore[return-value]
        async with self._login_lock:
            # Re-check after acquiring lock — another waiter may have refreshed.
            if not self._is_token_valid():
                await self._login()
        return self._token  # type: ignore[return-value]

    def is_authenticated(self) -> bool:
        """Synchronous liveness check — True when a non-expired token is cached."""
        return self._is_token_valid()

    async def start_auto_refresh(self, interval_minutes: int = 60) -> None:
        """Start a background task that proactively refreshes the token.

        The loop checks every ``interval_minutes`` and re-authenticates when
        the cached token is within ``_REFRESH_BUFFER`` (30 min) of expiry.
        For ProjectX's 24-hour tokens the default interval of 60 min is safe
        (vs TradeStationAuthV3's 10 min default, which targets ~30-min tokens).

        Args:
            interval_minutes: How often the background loop wakes up to check.
                              Must be > 0.
        """
        if interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be > 0, got {interval_minutes}")
        if self._refresh_task is not None:
            logger.warning("ProjectX auto-refresh is already running")
            return
        self._should_stop = False
        self._refresh_task = asyncio.create_task(self._refresh_loop(interval_minutes))
        logger.info(f"ProjectX auto-refresh started (interval={interval_minutes} min)")

    async def cleanup(self) -> None:
        """Cancel the background refresh task and close the HTTP client."""
        self._should_stop = True
        self._closed = True
        if self._refresh_task is not None:
            task = self._refresh_task
            self._refresh_task = None  # clear before awaiting to prevent re-entry
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        logger.info("ProjectX auth cleaned up")

    async def validate_session(self) -> bool:
        """Call the server-side validate endpoint to confirm the token is live.

        Returns:
            True if the server accepts the current token, False otherwise.
        """
        if not self._token:
            return False
        try:
            client = await self._get_client()
            resp = await client.post(
                self.VALIDATE_ENDPOINT,
                headers=self._bearer_headers(),
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"ProjectX validate_session failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _login(self) -> None:
        """Exchange API key for JWT; retry up to len(_RETRY_DELAYS) times."""
        client = await self._get_client()
        last_exc: Optional[Exception] = None

        for attempt, delay in enumerate(_RETRY_DELAYS):
            try:
                resp = await client.post(
                    self.AUTH_ENDPOINT,
                    json={"userName": self._username, "apiKey": self._api_key},
                    headers={"Content-Type": "application/json", "Accept": "application/json"},
                )
                resp.raise_for_status()
                token = _extract_token(resp.json())
                if not token:
                    raise ProjectXAuthError(
                        f"No token field in response (status {resp.status_code})"
                    )
                self._token = token
                self._token_expires_at = (
                    _decode_jwt_exp(token)
                    or datetime.now(tz=timezone.utc) + timedelta(hours=23)
                )
                logger.info(
                    f"ProjectX auth OK | hash={_token_hash(token)} | "
                    f"expires={self._token_expires_at.strftime('%Y-%m-%d %H:%M UTC')}"
                )
                return

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # Never log response body on 4xx — may echo credentials.
                logger.warning(
                    f"ProjectX login attempt {attempt + 1} → HTTP {exc.response.status_code}"
                )
                # 4xx (except 429 rate-limit) are permanent — do not retry.
                if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise ProjectXAuthError(
                        f"ProjectX login rejected (HTTP {exc.response.status_code})"
                    ) from exc

            except ProjectXAuthError:
                raise

            except Exception as exc:
                last_exc = exc
                logger.warning(f"ProjectX login attempt {attempt + 1} exception: {exc}")

            if attempt < len(_RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)

        raise ProjectXAuthError(
            f"ProjectX login failed after {len(_RETRY_DELAYS)} attempts"
        ) from last_exc

    async def _refresh_loop(self, interval_minutes: int) -> None:
        # Refresh immediately if token is already stale when the loop starts.
        if not self._is_token_valid():
            try:
                await self._login()
            except Exception as exc:
                logger.error(f"ProjectX initial refresh failed: {exc}")

        interval_secs = interval_minutes * 60
        while not self._should_stop:
            try:
                await asyncio.sleep(interval_secs)
                if self._should_stop:
                    break
                if not self._is_token_valid():
                    logger.info("ProjectX token nearing expiry — re-authenticating")
                    await self._login()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"ProjectX auto-refresh error: {exc}")
        logger.info("ProjectX auto-refresh loop exited")

    def _is_token_valid(self) -> bool:
        if not self._token:
            return False
        if self._token_expires_at is None:
            # Expiry unknown — assume valid (avoids hammering the endpoint).
            return True
        return datetime.now(tz=timezone.utc) < (self._token_expires_at - _REFRESH_BUFFER)

    def _bearer_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise ProjectXAuthError("ProjectXAuth has been cleaned up — create a new instance.")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _extract_token(data: dict) -> Optional[str]:
    """Pull the JWT string from common ProjectX response shapes."""
    for key in ("token", "accessToken", "access_token", "jwt"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    # Nested under "data" envelope
    nested = data.get("data")
    if isinstance(nested, dict):
        for key in ("token", "accessToken", "access_token", "jwt"):
            val = nested.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _decode_jwt_exp(token: str) -> Optional[datetime]:
    """Extract the ``exp`` claim from a JWT payload without verifying the signature."""
    try:
        payload_b64 = token.split(".")[1]
        rem = len(payload_b64) % 4
        if rem:
            payload_b64 += "=" * (4 - rem)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(int(exp), tz=timezone.utc)
    except Exception as exc:
        logger.debug(f"JWT exp decode failed (using 23h fallback): {exc}")
    return None


def _token_hash(token: str) -> str:
    """Return first 16 hex chars of SHA-256(token) for safe log output."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]
