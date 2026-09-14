"""Provider-neutral playlist ingestion entry point."""

from app.playlist.exportify import ExportifyCSVSource
from app.playlist.models import Playlist
from app.playlist.source import PlaylistInput, PlaylistSource
from app.playlist.spotify import SpotifyPlaylistReference, parse_spotify_playlist_url

__all__ = [
    "SpotifyPlaylistReference",
    "parse_spotify_playlist_url",
    "resolve_playlist",
]


def resolve_playlist(
    playlist_input: PlaylistInput,
    source: PlaylistSource | None = None,
) -> Playlist:
    """Resolve input through the selected source, defaulting to Exportify CSV."""
    playlist_source = source if source is not None else ExportifyCSVSource()
    return playlist_source.resolve_playlist(playlist_input)
