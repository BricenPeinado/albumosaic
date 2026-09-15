"""Tests for in-memory Spotify PKCE authentication."""

from __future__ import annotations

import logging
from io import BytesIO
from json import dumps
from re import fullmatch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from app.playlist import spotify_auth
from app.playlist.spotify_auth import (
    SpotifyAuthenticationError,
    SpotifyConfigurationError,
    SpotifyOAuthConfig,
    SpotifyOAuthManager,
    SpotifyToken,
)


class JsonResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self.payload = dumps(payload).encode()
        self.status = status

    def __enter__(self) -> JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload

    def getcode(self) -> int:
        return self.status


class CallbackRequest:
    """Minimal socket-like request used by BaseHTTPRequestHandler."""

    def __init__(self, path: str) -> None:
        self.input = BytesIO(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
        self.output = BytesIO()

    def makefile(self, mode: str, buffering: int | None = None) -> BytesIO:
        del buffering
        assert mode == "rb"
        return self.input

    def sendall(self, data: bytes) -> None:
        self.output.write(data)


def _request_form(request: Request) -> dict[str, list[str]]:
    data = request.data
    assert isinstance(data, bytes)
    return parse_qs(data.decode("ascii"))


def test_pkce_callback_exchanges_code_and_validates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> JsonResponse:
        del timeout
        requests.append(request)
        return JsonResponse(
            {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}
        )

    monkeypatch.setattr(spotify_auth, "open_url", fake_urlopen)
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    authorization_url = manager.begin_authorization()
    query = parse_qs(urlparse(authorization_url).query)
    state = query["state"][0]
    original_verifier = manager._pending_authorizations[state].code_verifier

    assert query["code_challenge_method"] == ["S256"]
    assert "code_challenge" in query
    manager.complete_callback(f"/spotify/callback?code=abc&state={state}")
    assert manager.access_token() == "access"
    assert manager.connection_status == "Connected to Spotify"
    assert len(requests) == 1

    request = requests[0]
    form = _request_form(request)
    assert request.full_url == "https://accounts.spotify.com/api/token"
    assert request.method == "POST"
    assert request.get_header("Content-type") == ("application/x-www-form-urlencoded")
    assert form == {
        "grant_type": ["authorization_code"],
        "code": ["abc"],
        "redirect_uri": [query["redirect_uri"][0]],
        "client_id": ["client"],
        "code_verifier": [original_verifier],
    }


def test_generated_verifier_is_rfc_7636_compatible() -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    query = parse_qs(urlparse(manager.begin_authorization()).query)
    verifier = manager._pending_authorizations[query["state"][0]].code_verifier

    assert 43 <= len(verifier) <= 128
    assert fullmatch(r"[A-Za-z0-9._~-]+", verifier)
    assert query["code_challenge"] == [spotify_auth._pkce_challenge(verifier)]


def test_pkce_challenge_matches_rfc_7636_vector() -> None:
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

    assert spotify_auth._pkce_challenge(verifier) == (
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_invalid_oauth_state_is_rejected() -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    manager.begin_authorization()

    with pytest.raises(SpotifyAuthenticationError, match="OAuth state"):
        manager.complete_callback("/spotify/callback?code=abc&state=wrong")


def test_missing_verifier_is_rejected_before_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    query = parse_qs(urlparse(manager.begin_authorization()).query)
    state = query["state"][0]
    manager._pending_authorizations[state] = spotify_auth._PendingAuthorization(
        code_verifier="",
        redirect_uri=query["redirect_uri"][0],
    )
    calls = 0

    def unexpected_urlopen(request: object, timeout: float) -> JsonResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return JsonResponse({})

    monkeypatch.setattr(spotify_auth, "open_url", unexpected_urlopen)

    with pytest.raises(SpotifyAuthenticationError, match="no associated PKCE verifier"):
        manager.complete_callback(f"/spotify/callback?code=abc&state={state}")
    assert calls == 0


def test_state_and_authorization_code_cannot_be_exchanged_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> JsonResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return JsonResponse({"access_token": "access", "expires_in": 3600})

    monkeypatch.setattr(spotify_auth, "open_url", fake_urlopen)
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    query = parse_qs(urlparse(manager.begin_authorization()).query)
    callback = f"/spotify/callback?code=single-use&state={query['state'][0]}"

    manager.complete_callback(callback)
    with pytest.raises(SpotifyAuthenticationError, match="Invalid Spotify OAuth state"):
        manager.complete_callback(callback)

    assert calls == 1


@pytest.mark.parametrize(
    ("oauth_error", "description", "expected"),
    [
        (
            "invalid_grant",
            "Invalid authorization code",
            "Spotify token request failed (invalid_grant): Invalid authorization code",
        ),
        (
            "invalid_client",
            "Invalid client",
            "Spotify token request failed (invalid_client): Invalid client",
        ),
    ],
)
def test_spotify_oauth_error_details_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    oauth_error: str,
    description: str,
    expected: str,
) -> None:
    error = HTTPError(
        spotify_auth._TOKEN_URL,
        400,
        "Bad Request",
        hdrs=None,
        fp=BytesIO(
            dumps({"error": oauth_error, "error_description": description}).encode()
        ),
    )
    monkeypatch.setattr(
        spotify_auth,
        "open_url",
        lambda request, timeout: (_ for _ in ()).throw(error),
    )
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    query = parse_qs(urlparse(manager.begin_authorization()).query)

    with pytest.raises(SpotifyAuthenticationError) as raised:
        manager.complete_callback(
            f"/spotify/callback?code=abc&state={query['state'][0]}"
        )
    assert str(raised.value) == expected


def test_callback_server_preserves_token_exchange_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    callback_request: CallbackRequest | None = None

    class CallbackServer:
        def __init__(self, address: object, handler: object) -> None:
            del address
            self.handler = handler
            self.timeout = 0.0
            self.closed = False

        def handle_request(self) -> None:
            nonlocal callback_request
            state = next(iter(manager._pending_authorizations))
            callback_request = CallbackRequest(
                f"/spotify/callback?code=abc&state={state}"
            )
            self.handler(callback_request, ("127.0.0.1", 12345), self)

        def server_close(self) -> None:
            self.closed = True

    token_error = HTTPError(
        spotify_auth._TOKEN_URL,
        400,
        "Bad Request",
        hdrs=None,
        fp=BytesIO(
            dumps(
                {
                    "error": "invalid_grant",
                    "error_description": "Invalid authorization code",
                }
            ).encode()
        ),
    )
    monkeypatch.setattr(spotify_auth, "HTTPServer", CallbackServer)
    monkeypatch.setattr(spotify_auth, "open_browser", lambda url: True)
    monkeypatch.setattr(
        spotify_auth,
        "open_url",
        lambda request, timeout: (_ for _ in ()).throw(token_error),
    )

    with pytest.raises(SpotifyAuthenticationError) as raised:
        manager.connect(callback_timeout=1)

    assert str(raised.value) == (
        "Spotify token request failed (invalid_grant): Invalid authorization code"
    )
    assert manager._pending_authorizations == {}
    assert callback_request is not None
    assert b"HTTP/1.0 400 Bad Request" in callback_request.output.getvalue()


def test_debug_diagnostics_are_safe(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        spotify_auth,
        "open_url",
        lambda request, timeout: JsonResponse(
            {"access_token": "secret-access", "expires_in": 3600}
        ),
    )
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client-id-ending-1234"))

    with caplog.at_level(logging.DEBUG, logger=spotify_auth.__name__):
        query = parse_qs(urlparse(manager.begin_authorization()).query)
        manager.complete_callback(
            f"/spotify/callback?code=secret-code&state={query['state'][0]}"
        )

    messages = caplog.text
    assert "client_id_present=True" in messages
    assert "client_id_suffix=1234" in messages
    assert "authorize_redirect_uri=http://127.0.0.1:8888/spotify/callback" in messages
    assert "token_redirect_uri=http://127.0.0.1:8888/spotify/callback" in messages
    assert "verifier_found=True" in messages
    assert "verifier_length=86" in messages
    assert "Spotify token endpoint HTTP status: 200" in messages
    assert "client-id-ending-1234" not in messages
    assert "secret-code" not in messages
    assert "secret-access" not in messages


def test_expired_token_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    manager._token = SpotifyToken("expired", "refresh", 0)
    monkeypatch.setattr(
        spotify_auth,
        "open_url",
        lambda request, timeout: JsonResponse(
            {"access_token": "fresh", "expires_in": 3600}
        ),
    )

    assert manager.access_token() == "fresh"
    assert manager._token is not None
    assert manager._token.refresh_token == "refresh"


def test_missing_environment_configuration_is_graceful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)

    with pytest.raises(SpotifyConfigurationError, match="SPOTIFY_CLIENT_ID"):
        SpotifyOAuthConfig.from_environment()


def test_redirect_uri_must_match_the_registered_callback_exactly() -> None:
    with pytest.raises(SpotifyConfigurationError, match="must be exactly"):
        SpotifyOAuthConfig("client", "http://localhost:8888/spotify/callback")

    with pytest.raises(SpotifyConfigurationError, match="must be exactly"):
        SpotifyOAuthConfig("client", "http://127.0.0.1:8889/spotify/callback")
