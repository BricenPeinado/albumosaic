"""Pure parsing for Spotify playlist references; performs no network I/O."""

from dataclasses import dataclass
from re import fullmatch
from urllib.parse import unquote, urlparse


class SpotifyPlaylistURLParseError(ValueError):
    """Raised when text is not a valid Spotify playlist URL."""


@dataclass(frozen=True, slots=True)
class SpotifyPlaylistReference:
    """A validated playlist ID and its canonical open.spotify.com URL."""

    playlist_id: str
    source_url: str


def parse_spotify_playlist_url(value: str) -> SpotifyPlaylistReference:
    """Parse a Spotify playlist URL without fetching Spotify content."""
    candidate = value.strip()
    if not candidate:
        raise SpotifyPlaylistURLParseError("Spotify playlist URL cannot be empty")

    try:
        parsed = urlparse(candidate)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise SpotifyPlaylistURLParseError("Malformed Spotify playlist URL") from error

    if parsed.scheme.casefold() not in {"http", "https"}:
        raise SpotifyPlaylistURLParseError(
            "Spotify playlist URL must use HTTP or HTTPS"
        )
    if hostname is None or hostname.casefold() != "open.spotify.com":
        raise SpotifyPlaylistURLParseError(
            "Spotify playlist URL must use the open.spotify.com host"
        )
    if parsed.username is not None or parsed.password is not None or port is not None:
        raise SpotifyPlaylistURLParseError(
            "Spotify playlist URL cannot contain credentials or a custom port"
        )

    path_parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(path_parts) != 2:
        raise SpotifyPlaylistURLParseError(
            "Spotify playlist URL must have the path /playlist/PLAYLIST_ID"
        )
    resource_type, playlist_id = path_parts
    if resource_type.casefold() != "playlist":
        raise SpotifyPlaylistURLParseError(
            f"Expected a Spotify playlist URL, received {resource_type or 'unknown'}"
        )
    if fullmatch(r"[A-Za-z0-9]+", playlist_id) is None:
        raise SpotifyPlaylistURLParseError(
            "Spotify playlist ID must be a non-empty base-62 identifier"
        )

    return SpotifyPlaylistReference(
        playlist_id=playlist_id,
        source_url=f"https://open.spotify.com/playlist/{playlist_id}",
    )
