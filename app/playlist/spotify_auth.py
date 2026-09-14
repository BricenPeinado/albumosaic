"""In-memory Spotify Authorization Code with PKCE authentication."""

from __future__ import annotations

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
        parsed = urlparse(self.redirect_uri)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path != "/spotify/callback"
            or parsed.query
            or parsed.fragment
        ):
            raise SpotifyConfigurationError(
                "SPOTIFY_REDIRECT_URI must be an HTTP 127.0.0.1 callback ending "
                "in /spotify/callback and containing an explicit port"
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


class SpotifyOAuthManager:
    """Create PKCE authorization requests and refresh in-memory tokens."""

    def __init__(self, config: SpotifyOAuthConfig, *, timeout: float = 15.0) -> None:
        if timeout <= 0:
            raise ValueError("Spotify OAuth timeout must be positive")
        self.config = config
        self.timeout = timeout
        self._token: SpotifyToken | None = None
        self._state: str | None = None
        self._verifier: str | None = None
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
        challenge = urlsafe_b64encode(sha256(verifier.encode()).digest()).rstrip(b"=")
        with self._lock:
            self._state = state
            self._verifier = verifier
        query = urlencode(
            {
                "client_id": self.config.client_id,
                "response_type": "code",
                "redirect_uri": self.config.redirect_uri,
                "state": state,
                "scope": _SCOPES,
                "code_challenge_method": "S256",
                "code_challenge": challenge.decode("ascii"),
            }
        )
        return f"{_AUTHORIZE_URL}?{query}"

    def connect(self, *, callback_timeout: float = 180.0) -> None:
        """Open Spotify authorization and receive one loopback callback."""
        if callback_timeout <= 0:
            raise ValueError("Spotify callback timeout must be positive")
        authorization_url = self.begin_authorization()
        parsed = urlparse(self.config.redirect_uri)
        manager = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                try:
                    manager.complete_callback(self.path)
                except SpotifyAuthenticationError as error:
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
                    f"Could not open a browser. Open this URL manually: {authorization_url}"
                )
            server.handle_request()
        finally:
            server.server_close()
        if not self.is_connected:
            raise SpotifyAuthenticationError(
                "Spotify authorization timed out or failed"
            )

    def complete_callback(self, callback_path: str) -> None:
        """Validate an OAuth callback and exchange its code for a token."""
        parsed = urlparse(callback_path)
        if parsed.path != "/spotify/callback":
            raise SpotifyAuthenticationError("Invalid Spotify callback path")
        values = parse_qs(parsed.query)
        if "error" in values:
            raise SpotifyAuthenticationError(
                f"Spotify authorization was denied: {values['error'][0]}"
            )
        state = values.get("state", [""])[0]
        code = values.get("code", [""])[0]
        with self._lock:
            expected_state = self._state
            verifier = self._verifier
        if not expected_state or not compare_digest(state, expected_state):
            raise SpotifyAuthenticationError("Invalid Spotify OAuth state")
        if not code or not verifier:
            raise SpotifyAuthenticationError("Spotify callback contained no code")
        payload = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
                "code_verifier": verifier,
            }
        )
        self._store_token(payload, previous_refresh_token=None)
        with self._lock:
            self._state = None
            self._verifier = None

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
            data=urlencode(form).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return _json_object(response.read())
        except HTTPError as error:
            raise SpotifyAuthenticationError(
                f"Spotify token request returned HTTP {error.code}"
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise SpotifyAuthenticationError("Spotify token request failed") from error

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
