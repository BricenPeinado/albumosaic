"""Tests for in-memory Spotify PKCE authentication."""

from __future__ import annotations

from json import dumps
from urllib.parse import parse_qs, urlparse

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
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = dumps(payload).encode()

    def __enter__(self) -> JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload


def test_pkce_callback_exchanges_code_and_validates_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> JsonResponse:
        del timeout
        requests.append(request)
        return JsonResponse(
            {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600}
        )

    monkeypatch.setattr(spotify_auth, "urlopen", fake_urlopen)
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    authorization_url = manager.begin_authorization()
    query = parse_qs(urlparse(authorization_url).query)

    assert query["code_challenge_method"] == ["S256"]
    assert "code_challenge" in query
    manager.complete_callback(f"/spotify/callback?code=abc&state={query['state'][0]}")
    assert manager.access_token() == "access"
    assert manager.connection_status == "Connected to Spotify"
    assert len(requests) == 1


def test_invalid_oauth_state_is_rejected() -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    manager.begin_authorization()

    with pytest.raises(SpotifyAuthenticationError, match="OAuth state"):
        manager.complete_callback("/spotify/callback?code=abc&state=wrong")


def test_expired_token_is_refreshed(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = SpotifyOAuthManager(SpotifyOAuthConfig("client"))
    manager._token = SpotifyToken("expired", "refresh", 0)
    monkeypatch.setattr(
        spotify_auth,
        "urlopen",
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


def test_redirect_uri_must_be_an_explicit_loopback_callback() -> None:
    with pytest.raises(SpotifyConfigurationError, match=r"127.0.0.1"):
        SpotifyOAuthConfig("client", "http://localhost:8888/spotify/callback")
