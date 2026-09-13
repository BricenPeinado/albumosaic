"""Spotify playlist ingestion boundary."""

from app.playlist.models import Playlist


def ingest_playlist(playlist_url: str) -> Playlist:
    """Load playlist metadata and its unique albums from Spotify.

    Authentication and Spotify API integration will be added in a later stage.
    """
    raise NotImplementedError("Spotify playlist ingestion is not implemented yet")
