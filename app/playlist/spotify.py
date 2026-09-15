"""Spotify playlist URL parsing and metadata-only Web API ingestion."""

from dataclasses import dataclass
from json import loads
from re import fullmatch
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request

from app.network import open_url
from app.playlist.models import Album, Playlist, Track
from app.playlist.source import PlaylistInput, PlaylistSource
from app.playlist.spotify_auth import SpotifyAuthenticationError, SpotifyOAuthManager


class SpotifyPlaylistURLParseError(ValueError):
    """Raised when text is not a valid Spotify playlist URL."""


class SpotifyPlaylistError(RuntimeError):
    """Raised when Spotify playlist metadata cannot be resolved."""


class SpotifyPlaylistAccessError(SpotifyPlaylistError):
    """Raised for Spotify's ownership or collaboration restriction."""


class SpotifyRateLimitError(SpotifyPlaylistError):
    """Raised when Spotify asks the client to retry later."""


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


class SpotifyPlaylistSource(PlaylistSource):
    """Resolve Spotify playlist metadata without retaining artwork URLs."""

    def __init__(self, oauth: SpotifyOAuthManager, *, timeout: float = 15.0) -> None:
        if timeout <= 0:
            raise ValueError("Spotify API timeout must be positive")
        self.oauth = oauth
        self.timeout = timeout

    @property
    def is_connected(self) -> bool:
        return self.oauth.is_connected

    @property
    def connection_status(self) -> str:
        return self.oauth.connection_status

    def connect(self) -> None:
        """Run the local browser-based PKCE flow."""
        self.oauth.connect()

    def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
        """Fetch every current ``/items`` page and map supported tracks."""
        if not isinstance(playlist_input, str):
            raise TypeError("The Spotify playlist input must be a URL")
        reference = parse_spotify_playlist_url(playlist_input)
        if not self.is_connected:
            raise SpotifyAuthenticationError("Connect Spotify first.")

        tracks: list[Track] = []
        offset = 0
        while True:
            query = urlencode(
                {"limit": 50, "offset": offset, "additional_types": "track"}
            )
            page = self._get_json(
                f"https://api.spotify.com/v1/playlists/"
                f"{reference.playlist_id}/items?{query}"
            )
            items = page.get("items")
            if not isinstance(items, list):
                raise SpotifyPlaylistError("Spotify returned invalid playlist items")
            for raw_item in items:
                track = _map_playlist_item(raw_item)
                if track is not None:
                    tracks.append(track)

            total = page.get("total")
            offset += len(items)
            if not items or (isinstance(total, int) and offset >= total):
                break
            next_url = page.get("next")
            if next_url is None and not isinstance(total, int):
                break

        return Playlist(
            playlist_id=reference.playlist_id,
            playlist_name=f"Spotify Playlist {reference.playlist_id}",
            source_url=reference.source_url,
            tracks=tuple(tracks),
        )

    def _get_json(self, url: str, *, refreshed: bool = False) -> dict[str, object]:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "api.spotify.com":
            raise SpotifyPlaylistError("Refused an unexpected Spotify API URL")
        request = Request(
            url,
            headers={"Authorization": f"Bearer {self.oauth.access_token()}"},
        )
        try:
            with open_url(request, timeout=self.timeout) as response:
                return _json_object(response.read())
        except HTTPError as error:
            if error.code == 401 and not refreshed:
                self.oauth.refresh()
                return self._get_json(url, refreshed=True)
            if error.code == 401:
                raise SpotifyAuthenticationError(
                    "Spotify authorization expired. Connect Spotify again."
                ) from error
            if error.code == 403:
                raise SpotifyPlaylistAccessError(
                    "Spotify currently allows Albumosaic to read playlist contents "
                    "only for playlists you own or collaborate on. Try one of your "
                    "own playlists, or use the Exportify CSV option."
                ) from error
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "unknown")
                raise SpotifyRateLimitError(
                    f"Spotify rate limit reached. Try again after {retry_after} seconds."
                ) from error
            raise SpotifyPlaylistError(
                f"Spotify playlist request returned HTTP {error.code}"
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise SpotifyPlaylistError("Spotify playlist request failed") from error


def _map_playlist_item(raw_item: object) -> Track | None:
    if not isinstance(raw_item, dict):
        return None
    item = raw_item.get("item")
    if not isinstance(item, dict) or item.get("type") != "track":
        return None
    album_data = item.get("album")
    if not isinstance(album_data, dict):
        return None
    track_name = _text(item.get("name"))
    album_name = _text(album_data.get("name"))
    track_artists = _artist_names(item.get("artists"))
    album_artists = _artist_names(album_data.get("artists")) or track_artists
    if not track_name or not album_name or not track_artists or not album_artists:
        return None

    album = Album(
        album_id=_text(album_data.get("id")),
        album_name=album_name,
        artists=album_artists,
        source_url=_spotify_external_url(album_data),
        release_date=_text(album_data.get("release_date")),
    )
    external_ids = item.get("external_ids")
    return Track(
        track_id=_text(item.get("id")),
        track_name=track_name,
        artists=track_artists,
        album=album,
        source_url=_spotify_external_url(item),
        disc_number=_integer(item.get("disc_number")),
        track_number=_integer(item.get("track_number")),
        duration_ms=_integer(item.get("duration_ms")),
        preview_url=None,
        explicit=item.get("explicit")
        if isinstance(item.get("explicit"), bool)
        else None,
        popularity=_integer(item.get("popularity")),
        isrc=(
            _text(external_ids.get("isrc")) if isinstance(external_ids, dict) else None
        ),
        added_by=_added_by(raw_item.get("added_by")),
        added_at=_text(raw_item.get("added_at")),
    )


def _artist_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        name
        for entry in value
        if isinstance(entry, dict) and (name := _text(entry.get("name"))) is not None
    )


def _spotify_external_url(value: dict[object, object]) -> str | None:
    external_urls = value.get("external_urls")
    if not isinstance(external_urls, dict):
        return None
    return _text(external_urls.get("spotify"))


def _added_by(value: object) -> str | None:
    return _text(value.get("id")) if isinstance(value, dict) else None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _json_object(payload: bytes) -> dict[str, object]:
    try:
        decoded = loads(payload)
    except (UnicodeDecodeError, ValueError) as error:
        raise SpotifyPlaylistError("Spotify returned malformed JSON") from error
    if not isinstance(decoded, dict):
        raise SpotifyPlaylistError("Spotify returned an invalid JSON response")
    return cast(dict[str, object], decoded)
