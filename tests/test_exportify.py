"""Tests for provider-neutral Exportify CSV playlist ingestion."""

from csv import DictWriter
from io import StringIO
from pathlib import Path

import pytest

from app.playlist.exportify import ExportifyCSVError, ExportifyCSVSource
from app.playlist.models import Playlist
from app.playlist.parser import resolve_playlist
from app.playlist.source import PlaylistInput, PlaylistSource

EXPORTIFY_HEADERS = [
    "Track URI",
    "Track Name",
    "Artist Name(s)",
    "Album URI",
    "Album Name",
    "Album Artist Name(s)",
    "Album Release Date",
    "Album Type",
    "Album Image URL",
    "Disc Number",
    "Track Number",
    "Track Duration (ms)",
    "Track Preview URL",
    "Explicit?",
    "Popularity",
    "ISRC",
    "Added By",
    "Added At",
]


def csv_stream(rows: list[dict[str, str]]) -> StringIO:
    stream = StringIO(newline="")
    writer = DictWriter(stream, fieldnames=EXPORTIFY_HEADERS)
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def exportify_row(
    *,
    track_id: str,
    track_name: str,
    track_artists: str,
    album_id: str,
    album_name: str,
    album_artists: str,
) -> dict[str, str]:
    return {
        "Track URI": f"spotify:track:{track_id}" if track_id else "",
        "Track Name": track_name,
        "Artist Name(s)": track_artists,
        "Album URI": f"spotify:album:{album_id}" if album_id else "",
        "Album Name": album_name,
        "Album Artist Name(s)": album_artists,
        "Album Release Date": "2001-01-01",
        "Album Type": "album",
        "Album Image URL": "https://images.example/cover.jpg",
        "Disc Number": "1",
        "Track Number": "2",
        "Track Duration (ms)": "245000",
        "Track Preview URL": "https://audio.example/preview.mp3",
        "Explicit?": "false",
        "Popularity": "87",
        "ISRC": "US-ABC-01-00001",
        "Added By": "spotify:user:listener",
        "Added At": "2026-09-12T00:00:00Z",
    }


def test_playlist_source_is_abstract() -> None:
    with pytest.raises(TypeError):
        PlaylistSource()


def test_exportify_extracts_tracks_albums_and_available_metadata() -> None:
    rows = [
        exportify_row(
            track_id="track-1",
            track_name="First Song",
            track_artists="Main Artist, Guest Artist",
            album_id="album-1",
            album_name="The Album",
            album_artists="Main Artist",
        ),
        exportify_row(
            track_id="track-2",
            track_name="Second Song",
            track_artists="Main Artist",
            album_id="album-1",
            album_name="The Album",
            album_artists="Main Artist",
        ),
    ]

    playlist = ExportifyCSVSource(playlist_name="My Export").resolve_playlist(
        csv_stream(rows)
    )

    assert playlist.playlist_name == "My Export"
    assert len(playlist.tracks) == 2
    assert playlist.unique_album_count == 1
    first_track = playlist.tracks[0]
    assert first_track.track_id == "track-1"
    assert first_track.artists == ("Main Artist", "Guest Artist")
    assert first_track.source_url == "https://open.spotify.com/track/track-1"
    assert first_track.duration_ms == 245000
    assert first_track.disc_number == 1
    assert first_track.track_number == 2
    assert first_track.explicit is False
    assert first_track.popularity == 87
    assert first_track.isrc == "US-ABC-01-00001"
    assert first_track.album.album_id == "album-1"
    assert first_track.album.album_name == "The Album"
    assert first_track.album.artists == ("Main Artist",)
    assert first_track.album.release_date == "2001-01-01"
    assert first_track.album.album_type == "album"
    assert first_track.album.album_id == "album-1"
    assert first_track.album.source_url == "https://open.spotify.com/album/album-1"


def test_exportify_fallback_deduplicates_case_but_not_different_artists() -> None:
    rows = [
        exportify_row(
            track_id="track-1",
            track_name="One",
            track_artists="Artist One",
            album_id="",
            album_name="Home",
            album_artists="Artist One",
        ),
        exportify_row(
            track_id="track-2",
            track_name="Two",
            track_artists="ARTIST ONE",
            album_id="",
            album_name="  HOME ",
            album_artists="ARTIST ONE",
        ),
        exportify_row(
            track_id="track-3",
            track_name="Three",
            track_artists="Artist Two",
            album_id="",
            album_name="Home",
            album_artists="Artist Two",
        ),
    ]

    playlist = ExportifyCSVSource().resolve_playlist(csv_stream(rows))

    assert len(playlist.tracks) == 3
    assert [album.artists for album in playlist.albums] == [
        ("Artist One",),
        ("Artist Two",),
    ]


def test_duplicate_csv_tracks_still_produce_one_unique_album() -> None:
    row = exportify_row(
        track_id="track-1",
        track_name="Repeated Song",
        track_artists="Artist",
        album_id="album-1",
        album_name="Album",
        album_artists="Artist",
    )

    playlist = ExportifyCSVSource().resolve_playlist(csv_stream([row, row]))

    assert len(playlist.tracks) == 2
    assert playlist.unique_album_count == 1


def test_exportify_file_uses_filename_as_playlist_name(tmp_path: Path) -> None:
    path = tmp_path / "Road Trip.csv"
    contents = csv_stream(
        [
            exportify_row(
                track_id="track-1",
                track_name="Song",
                track_artists="Artist",
                album_id="album-1",
                album_name="Album",
                album_artists="Artist",
            )
        ]
    ).getvalue()
    path.write_text("\ufeff" + contents, encoding="utf-8")

    playlist = resolve_playlist(path)

    assert playlist.playlist_name == "Road Trip"


def test_exportify_rejects_missing_required_columns() -> None:
    stream = StringIO("Track Name,Artist Name(s)\nSong,Artist\n")

    with pytest.raises(ExportifyCSVError, match="album_name"):
        ExportifyCSVSource().resolve_playlist(stream)


def test_resolve_playlist_delegates_to_selected_source() -> None:
    expected = Playlist(
        playlist_id=None,
        playlist_name="Stub",
        source_url=None,
        tracks=(),
    )

    class StubSource(PlaylistSource):
        def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
            assert playlist_input == "provider-specific-input"
            return expected

    assert resolve_playlist("provider-specific-input", StubSource()) is expected
