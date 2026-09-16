"""Typed playlist-domain models and deterministic album deduplication."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from unicodedata import normalize as unicode_normalize

AlbumKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Album:
    """Album metadata required to retrieve and identify mosaic artwork."""

    album_id: str | None
    album_name: str
    artists: tuple[str, ...]
    source_url: str | None
    release_date: str | None = None
    album_type: str | None = None

    @property
    def identity_key(self) -> AlbumKey:
        """Return a stable ID key or a normalized artist/title fallback."""
        if self.album_id and self.album_id.strip():
            return ("id", self.album_id.strip())
        return (
            "artist-title",
            _normalize_text(" ".join(self.artists)),
            _normalize_text(self.album_name),
        )


@dataclass(frozen=True, slots=True)
class Track:
    """A playlist track and its parent album."""

    track_id: str | None
    track_name: str
    artists: tuple[str, ...]
    album: Album
    source_url: str | None
    disc_number: int | None = None
    track_number: int | None = None
    duration_ms: int | None = None
    preview_url: str | None = None
    explicit: bool | None = None
    popularity: int | None = None
    isrc: str | None = None
    added_by: str | None = None
    added_at: str | None = None


@dataclass(frozen=True, slots=True)
class Playlist:
    """A playlist whose album collection is always deduplicated."""

    playlist_id: str | None
    playlist_name: str
    source_url: str | None
    tracks: tuple[Track, ...]
    albums: tuple[Album, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "albums",
            deduplicate_albums(track.album for track in self.tracks),
        )

    @property
    def unique_album_count(self) -> int:
        """Return the number of albums safe to pass to the mosaic pipeline."""
        return len(self.albums)


def deduplicate_albums(albums: Iterable[Album]) -> tuple[Album, ...]:
    """Return the first album for each stable or normalized identity key."""
    unique_albums: list[Album] = []
    seen_keys: set[AlbumKey] = set()
    for album in albums:
        identity_key = album.identity_key
        if identity_key not in seen_keys:
            seen_keys.add(identity_key)
            unique_albums.append(album)
    return tuple(unique_albums)


def _normalize_text(value: str) -> str:
    """Normalize Unicode, capitalization, and internal whitespace."""
    normalized = unicode_normalize("NFKC", value).casefold()
    return " ".join(normalized.split())
