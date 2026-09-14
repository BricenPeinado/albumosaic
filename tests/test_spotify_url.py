"""Tests for pure Spotify playlist URL parsing."""

import pytest

from app.playlist.spotify import (
    SpotifyPlaylistReference,
    SpotifyPlaylistURLParseError,
    parse_spotify_playlist_url,
)

PLAYLIST_ID = "3cEYpjA9oz9GiPac4AsH4n"


@pytest.mark.parametrize(
    "url",
    [
        f"https://open.spotify.com/playlist/{PLAYLIST_ID}",
        f"http://open.spotify.com/playlist/{PLAYLIST_ID}",
        f"https://OPEN.SPOTIFY.COM/playlist/{PLAYLIST_ID}/",
        f"  https://open.spotify.com/playlist/{PLAYLIST_ID}  ",
    ],
)
def test_valid_spotify_playlist_urls(url: str) -> None:
    assert parse_spotify_playlist_url(url) == SpotifyPlaylistReference(
        playlist_id=PLAYLIST_ID,
        source_url=f"https://open.spotify.com/playlist/{PLAYLIST_ID}",
    )


def test_url_with_query_string_and_fragment() -> None:
    reference = parse_spotify_playlist_url(
        f"https://open.spotify.com/playlist/{PLAYLIST_ID}?si=abc123#details"
    )

    assert reference.playlist_id == PLAYLIST_ID
    assert reference.source_url == f"https://open.spotify.com/playlist/{PLAYLIST_ID}"


@pytest.mark.parametrize(
    "url",
    [
        "not a URL",
        f"open.spotify.com/playlist/{PLAYLIST_ID}",
        f"https://example.com/playlist/{PLAYLIST_ID}",
        f"https://open.spotify.com.evil.test/playlist/{PLAYLIST_ID}",
        "https://open.spotify.com/playlist/",
        f"https://open.spotify.com/playlist/{PLAYLIST_ID}/extra",
        "https://open.spotify.com/playlist/not_a_base62_id",
        f"ftp://open.spotify.com/playlist/{PLAYLIST_ID}",
        f"https://user@open.spotify.com/playlist/{PLAYLIST_ID}",
        f"https://open.spotify.com:8443/playlist/{PLAYLIST_ID}",
        f"https://open.spotify.com:notaport/playlist/{PLAYLIST_ID}",
    ],
)
def test_malformed_urls_are_rejected(url: str) -> None:
    with pytest.raises(SpotifyPlaylistURLParseError):
        parse_spotify_playlist_url(url)


def test_album_link_is_rejected() -> None:
    with pytest.raises(SpotifyPlaylistURLParseError, match="album"):
        parse_spotify_playlist_url(f"https://open.spotify.com/album/{PLAYLIST_ID}")


def test_track_link_is_rejected() -> None:
    with pytest.raises(SpotifyPlaylistURLParseError, match="track"):
        parse_spotify_playlist_url(f"https://open.spotify.com/track/{PLAYLIST_ID}")


@pytest.mark.parametrize("value", ["", " ", "\n\t"])
def test_empty_input_is_rejected(value: str) -> None:
    with pytest.raises(SpotifyPlaylistURLParseError, match="empty"):
        parse_spotify_playlist_url(value)
