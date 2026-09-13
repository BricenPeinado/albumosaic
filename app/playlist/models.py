"""Domain models used by playlist and artwork services."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Album:
    """A unique album discovered in a Spotify playlist."""

    spotify_id: str
    name: str
    artist_names: tuple[str, ...]
    artwork_url: str | None


@dataclass(frozen=True, slots=True)
class Playlist:
    """The playlist metadata needed by the mosaic pipeline."""

    spotify_id: str
    name: str
    albums: tuple[Album, ...]

    @property
    def unique_album_count(self) -> int:
        """Return the number of deduplicated albums in the playlist."""
        return len(self.albums)
