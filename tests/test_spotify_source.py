"""Tests for Spotify's metadata-only playlist source."""

from __future__ import annotations

from email.message import Message
from json import dumps
from urllib.error import HTTPError

import pytest

from app.playlist import spotify
from app.playlist.spotify import (
    SpotifyPlaylistAccessError,
    SpotifyPlaylistSource,
    SpotifyRateLimitError,
)
from app.playlist.spotify_auth import SpotifyAuthenticationError


class FakeOAuth:
    is_connected = True
    connection_status = "Connected to Spotify"

    def __init__(self) -> None:
        self.refreshes = 0

    def access_token(self) -> str:
        return "secret-token"

    def refresh(self) -> None:
        self.refreshes += 1

    def connect(self) -> None:
        pass


class JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = dumps(payload).encode()

    def __enter__(self) -> JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload


def track_item(track_id: str, album_id: str = "album-1") -> dict[str, object]:
    return {
        "added_at": "2026-01-01T00:00:00Z",
        "item": {
            "type": "track",
            "id": track_id,
            "name": f"Track {track_id}",
            "artists": [{"name": "Artist"}],
            "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
            "album": {
                "id": album_id,
                "name": "Album",
                "album_type": "album",
                "release_date": "2001-01-01",
                "artists": [{"name": "Artist"}],
                "images": [{"url": "https://i.scdn.co/forbidden.jpg"}],
                "external_urls": {
                    "spotify": f"https://open.spotify.com/album/{album_id}"
                },
            },
        },
    }


def test_spotify_resolves_all_pages_deduplicates_and_ignores_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = iter(
        [
            {"items": [track_item("one")], "total": 2, "next": "present"},
            {"items": [track_item("two")], "total": 2, "next": None},
        ]
    )
    requested_urls: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> JsonResponse:
        del timeout
        requested_urls.append(request.full_url)  # type: ignore[attr-defined]
        return JsonResponse(next(pages))

    monkeypatch.setattr(spotify, "open_url", fake_urlopen)
    playlist = SpotifyPlaylistSource(FakeOAuth()).resolve_playlist(
        "https://open.spotify.com/playlist/abc123"
    )

    assert len(playlist.tracks) == 2
    assert playlist.unique_album_count == 1
    assert playlist.albums[0].album_id == "album-1"
    assert playlist.albums[0].album_type == "album"
    assert playlist.albums[0].release_date == "2001-01-01"
    assert all("/items?" in url for url in requested_urls)
    assert all("scdn.co" not in url for url in requested_urls)


def test_malformed_and_non_track_items_are_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = {
        "items": [None, {}, {"item": {"type": "episode"}}, track_item("valid")],
        "total": 4,
        "next": None,
    }
    monkeypatch.setattr(
        spotify, "open_url", lambda request, timeout: JsonResponse(page)
    )

    playlist = SpotifyPlaylistSource(FakeOAuth()).resolve_playlist(
        "https://open.spotify.com/playlist/abc123"
    )

    assert [track.track_id for track in playlist.tracks] == ["valid"]


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (403, SpotifyPlaylistAccessError, "own or collaborate"),
        (429, SpotifyRateLimitError, "17 seconds"),
    ],
)
def test_spotify_api_errors_are_user_friendly(
    status: int,
    error_type: type[Exception],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = Message()
    headers["Retry-After"] = "17"

    def fail(request: object, timeout: float) -> JsonResponse:
        del request, timeout
        raise HTTPError("url", status, "error", headers, None)

    monkeypatch.setattr(spotify, "open_url", fail)

    with pytest.raises(error_type, match=message):
        SpotifyPlaylistSource(FakeOAuth()).resolve_playlist(
            "https://open.spotify.com/playlist/abc123"
        )


def test_401_refreshes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    oauth = FakeOAuth()
    attempts = 0

    def response_after_refresh(request: object, timeout: float) -> JsonResponse:
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts == 1:
            raise HTTPError("url", 401, "unauthorized", Message(), None)
        return JsonResponse({"items": [], "total": 0, "next": None})

    monkeypatch.setattr(spotify, "open_url", response_after_refresh)
    SpotifyPlaylistSource(oauth).resolve_playlist(
        "https://open.spotify.com/playlist/abc123"
    )

    assert oauth.refreshes == 1


def test_repeated_401_requires_reconnection(monkeypatch: pytest.MonkeyPatch) -> None:
    oauth = FakeOAuth()

    def unauthorized(request: object, timeout: float) -> JsonResponse:
        del request, timeout
        raise HTTPError("url", 401, "unauthorized", Message(), None)

    monkeypatch.setattr(spotify, "open_url", unauthorized)

    with pytest.raises(SpotifyAuthenticationError, match="Connect Spotify again"):
        SpotifyPlaylistSource(oauth).resolve_playlist(
            "https://open.spotify.com/playlist/abc123"
        )
    assert oauth.refreshes == 1


def test_unconnected_source_requires_oauth() -> None:
    oauth = FakeOAuth()
    oauth.is_connected = False

    with pytest.raises(SpotifyAuthenticationError, match="Connect Spotify"):
        SpotifyPlaylistSource(oauth).resolve_playlist(
            "https://open.spotify.com/playlist/abc123"
        )
