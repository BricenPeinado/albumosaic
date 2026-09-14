"""In-memory Spotify Authorization Code with PKCE authentication."""

from __future__ import annotations

import logging
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import loads
from os import environ
from secrets import token_urlsafe
from threading import RLock
from time import time
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from webbrowser import open as open_browser

_AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
_TOKEN_URL = "https://accounts.spotify.com/api/token"
_DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/spotify/callback"
_SCOPES = "playlist-read-private playlist-read-collaborative"

logger = logging.getLogger(__name__)


class SpotifyConfigurationError(RuntimeError):
    """Raised when local Spotify application configuration is unavailable."""


class SpotifyAuthenticationError(RuntimeError):
    """Raised when Spotify authorization or token refresh fails."""


@dataclass(frozen=True, slots=True)
class SpotifyOAuthConfig:
    """Public PKCE client settings loaded without a client secret."""

    client_id: str
    redirect_uri: str = _DEFAULT_REDIRECT_URI

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise SpotifyConfigurationError("SPOTIFY_CLIENT_ID cannot be empty")
        if self.redirect_uri != _DEFAULT_REDIRECT_URI:
            raise SpotifyConfigurationError(
                f"SPOTIFY_REDIRECT_URI must be exactly {_DEFAULT_REDIRECT_URI}"
            )

    @classmethod
    def from_environment(cls) -> SpotifyOAuthConfig:
        """Load optional Spotify configuration from environment variables."""
        client_id = environ.get("SPOTIFY_CLIENT_ID", "").strip()
        if not client_id:
            raise SpotifyConfigurationError(
                "Spotify setup is missing. Set SPOTIFY_CLIENT_ID, then restart "
                "Albumosaic."
            )
        return cls(
            client_id=client_id,
            redirect_uri=environ.get(
                "SPOTIFY_REDIRECT_URI",
                _DEFAULT_REDIRECT_URI,
            ).strip(),
        )


@dataclass(frozen=True, slots=True)
class SpotifyToken:
    """An in-memory Spotify access token and refresh metadata."""

    access_token: str
    refresh_token: str | None
    expires_at: float


@dataclass(frozen=True, slots=True)
class _PendingAuthorization:
    """PKCE material retained for one OAuth state."""

    code_verifier: str
    redirect_uri: str


class SpotifyOAuthManager:
    """Create PKCE authorization requests and refresh in-memory tokens."""

    def __init__(self, config: SpotifyOAuthConfig, *, timeout: float = 15.0) -> None:
        if timeout <= 0:
            raise ValueError("Spotify OAuth timeout must be positive")
        self.config = config
        self.timeout = timeout
        self._token: SpotifyToken | None = None
        self._pending_authorizations: dict[str, _PendingAuthorization] = {}
        self._lock = RLock()

    @property
    def is_connected(self) -> bool:
        """Return whether the user has completed authorization."""
        return self._token is not None

    @property
    def connection_status(self) -> str:
        """Return a credential-free status suitable for the UI."""
        return "Connected to Spotify" if self.is_connected else "Connect Spotify first."

    def begin_authorization(self) -> str:
        """Create and retain a secure PKCE authorization request."""
        verifier = token_urlsafe(64)
        state = token_urlsafe(32)
        challenge = _pkce_challenge(verifier)
        with self._lock:
            self._pending_authorizations[state] = _PendingAuthorization(
                code_verifier=verifier,
                redirect_uri=self.config.redirect_uri,
            )
        logger.debug(
            "Starting Spotify PKCE authorization: client_id_present=%s, "
            "client_id_suffix=%s, authorize_redirect_uri=%s, verifier_length=%d",
            bool(self.config.client_id),
            self.config.client_id[-4:],
            self.config.redirect_uri,
            len(verifier),
        )
        query = urlencode(
            {
                "client_id": self.config.client_id,
                "response_type": "code",
                "redirect_uri": self.config.redirect_uri,
                "state": state,
                "scope": _SCOPES,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
        )
        return f"{_AUTHORIZE_URL}?{query}"

    def connect(self, *, callback_timeout: float = 180.0) -> None:
        """Open Spotify authorization and receive one loopback callback."""
        if callback_timeout <= 0:
            raise ValueError("Spotify callback timeout must be positive")
        authorization_url = self.begin_authorization()
        authorization_state = parse_qs(urlparse(authorization_url).query)["state"][0]
        parsed = urlparse(self.config.redirect_uri)
        manager = self
        callback_errors: list[SpotifyAuthenticationError] = []

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    manager.complete_callback(self.path)
                except SpotifyAuthenticationError as error:
                    callback_errors.append(error)
                    self.send_response(400)
                    message = f"Spotify connection failed: {error}".encode()
                else:
                    self.send_response(200)
                    message = b"Spotify connected. You can close this tab."
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(message)))
                self.end_headers()
                self.wfile.write(message)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        server = HTTPServer(
            (cast(str, parsed.hostname), cast(int, parsed.port)), CallbackHandler
        )
        server.timeout = callback_timeout
        try:
            if not open_browser(authorization_url):
                raise SpotifyAuthenticationError(
                    "Could not open a browser for Spotify authorization"
                )
            server.handle_request()
        finally:
            server.server_close()
            self._discard_authorization(authorization_state)
        if not self.is_connected:
            if callback_errors:
                raise callback_errors[0]
            raise SpotifyAuthenticationError(
                "Spotify authorization timed out or failed"
            )

    def complete_callback(self, callback_path: str) -> None:
        """Validate an OAuth callback and exchange its code for a token."""
        parsed = urlparse(callback_path)
        if parsed.path != "/spotify/callback":
            raise SpotifyAuthenticationError("Invalid Spotify callback path")
        values = parse_qs(parsed.query)
        state = values.get("state", [""])[0]
        code = values.get("code", [""])[0]
        authorization = self._consume_authorization(state)
        verifier_found = bool(authorization and authorization.code_verifier)
        logger.debug(
            "Spotify callback PKCE lookup: verifier_found=%s, verifier_length=%d",
            verifier_found,
            len(authorization.code_verifier) if authorization else 0,
        )
        if authorization is None:
            raise SpotifyAuthenticationError("Invalid Spotify OAuth state")
        if "error" in values:
            oauth_error = _safe_error_text(values["error"][0])
            raise SpotifyAuthenticationError(
                f"Spotify authorization was denied: {oauth_error or 'unknown error'}"
            )
        if not authorization.code_verifier:
            raise SpotifyAuthenticationError(
                "Spotify OAuth state has no associated PKCE verifier"
            )
        if not code:
            raise SpotifyAuthenticationError("Spotify callback contained no code")
        logger.debug(
            "Exchanging Spotify authorization code: token_redirect_uri=%s",
            authorization.redirect_uri,
        )
        payload = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": authorization.redirect_uri,
                "client_id": self.config.client_id,
                "code_verifier": authorization.code_verifier,
            }
        )
        self._store_token(payload, previous_refresh_token=None)

    def _consume_authorization(
        self, callback_state: str
    ) -> _PendingAuthorization | None:
        """Atomically validate and consume one pending OAuth state."""
        if not callback_state:
            return None
        with self._lock:
            matching_state = next(
                (
                    pending_state
                    for pending_state in self._pending_authorizations
                    if compare_digest(callback_state, pending_state)
                ),
                None,
            )
            if matching_state is None:
                return None
            return self._pending_authorizations.pop(matching_state)

    def _discard_authorization(self, state: str) -> None:
        """Remove unconsumed PKCE material after a callback server exits."""
        with self._lock:
            self._pending_authorizations.pop(state, None)

    def access_token(self) -> str:
        """Return a valid token, refreshing it shortly before expiration."""
        token = self._token
        if token is None:
            raise SpotifyAuthenticationError("Connect Spotify first.")
        if token.expires_at <= time() + 30:
            self.refresh()
            token = self._token
            assert token is not None
        return token.access_token

    def refresh(self) -> None:
        """Refresh the current PKCE token without exposing credentials."""
        token = self._token
        if token is None or not token.refresh_token:
            raise SpotifyAuthenticationError("Connect Spotify again.")
        payload = self._token_request(
            {
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
                "client_id": self.config.client_id,
            }
        )
        self._store_token(payload, previous_refresh_token=token.refresh_token)

    def _token_request(self, form: dict[str, str]) -> dict[str, object]:
        request = Request(
            _TOKEN_URL,
            data=urlencode(form).encode("ascii"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = response.getcode()
                logger.debug("Spotify token endpoint HTTP status: %d", status)
                payload = _json_object(response.read())
        except HTTPError as error:
            logger.debug("Spotify token endpoint HTTP status: %d", error.code)
            raise SpotifyAuthenticationError(_token_error_message(error)) from error
        except (TimeoutError, URLError, OSError) as error:
            detail = _safe_error_text(getattr(error, "reason", error))
            suffix = f": {detail}" if detail else ""
            raise SpotifyAuthenticationError(
                f"Spotify token request failed{suffix}"
            ) from error
        oauth_error = _oauth_error_message(payload)
        if oauth_error is not None:
            raise SpotifyAuthenticationError(oauth_error)
        return payload

    def _store_token(
        self,
        payload: dict[str, object],
        *,
        previous_refresh_token: str | None,
    ) -> None:
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        refresh_value = payload.get("refresh_token")
        if not isinstance(access_token, str) or not access_token:
            raise SpotifyAuthenticationError("Spotify returned no access token")
        if not isinstance(expires_in, int) or expires_in <= 0:
            raise SpotifyAuthenticationError(
                "Spotify returned an invalid token lifetime"
            )
        refresh_token = (
            refresh_value if isinstance(refresh_value, str) else previous_refresh_token
        )
        with self._lock:
            self._token = SpotifyToken(access_token, refresh_token, time() + expires_in)


def _json_object(payload: bytes) -> dict[str, object]:
    try:
        decoded = loads(payload)
    except (UnicodeDecodeError, ValueError) as error:
        raise SpotifyAuthenticationError("Spotify returned malformed JSON") from error
    if not isinstance(decoded, dict):
        raise SpotifyAuthenticationError("Spotify returned an invalid JSON response")
    return cast(dict[str, object], decoded)


def _pkce_challenge(verifier: str) -> str:
    """Return the RFC 7636 S256 challenge for a PKCE verifier."""
    digest = sha256(verifier.encode("ascii")).digest()
    return urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _token_error_message(error: HTTPError) -> str:
    """Extract only Spotify's safe OAuth error fields from an HTTP failure."""
    try:
        payload = _json_object(error.read())
    except SpotifyAuthenticationError:
        return f"Spotify token request failed (HTTP {error.code})"
    return _oauth_error_message(payload) or (
        f"Spotify token request failed (HTTP {error.code})"
    )


def _oauth_error_message(payload: dict[str, object]) -> str | None:
    oauth_error = _safe_error_text(payload.get("error"))
    if not oauth_error:
        return None
    description = _safe_error_text(payload.get("error_description"))
    message = f"Spotify token request failed ({oauth_error})"
    return f"{message}: {description}" if description else message


def _safe_error_text(value: object, *, maximum_length: int = 300) -> str:
    """Bound and normalize a non-secret OAuth or transport error value."""
    if not isinstance(value, (str, OSError)):
        return ""
    text = " ".join(str(value).split())
    return text[:maximum_length]
