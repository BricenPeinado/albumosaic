"""Tests for the initial playlist domain models."""

from app.playlist.models import Album, Playlist


def test_playlist_reports_unique_album_count() -> None:
    album = Album(
        spotify_id="album-1",
        name="Example Album",
        artist_names=("Example Artist",),
        artwork_url=None,
    )
    playlist = Playlist(
        spotify_id="playlist-1",
        name="Example Playlist",
        albums=(album,),
    )

    assert playlist.unique_album_count == 1
