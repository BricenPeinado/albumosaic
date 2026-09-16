"""Tests for provider-neutral UI workflow orchestration."""

import logging
from pathlib import Path

import pytest
from PIL import Image

from app.mosaic.matcher import MatchMode
from app.playlist.artwork_sources import ResolvedArtwork
from app.playlist.models import Album, Playlist, Track
from app.playlist.progress import (
    PreparationProgress,
    PreparationProgressReporter,
    PreparationStage,
)
from app.playlist.source import PlaylistInput, PlaylistSource
from app.playlist.spotify import SpotifyPlaylistSource
from app.workflow import (
    AlbumosaicWorkflow,
    SpotifyPlaylistResolutionUnavailable,
    UnconfiguredSpotifyPlaylistSource,
    WorkflowProgress,
    WorkflowStage,
    default_spotify_playlist_source,
)


def _playlist() -> Playlist:
    tracks = []
    for index in range(2):
        album = Album(
            album_id=f"album-{index}",
            album_name=f"Album {index}",
            artists=("Artist",),
            source_url=None,
        )
        tracks.append(
            Track(
                track_id=f"track-{index}",
                track_name=f"Track {index}",
                artists=("Artist",),
                album=album,
                source_url=None,
            )
        )
    return Playlist(None, "Test", None, tuple(tracks))


class _PlaylistSource(PlaylistSource):
    def __init__(self, playlist: Playlist) -> None:
        self.playlist = playlist

    def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
        assert playlist_input == "playlist-input"
        return self.playlist


class _ArtworkProvider:
    def __init__(self, paths: tuple[Path, ...]) -> None:
        self.paths = paths

    def get_many(
        self,
        albums: tuple[Album, ...],
        progress_reporter: PreparationProgressReporter | None = None,
    ) -> tuple[ResolvedArtwork, ...]:
        assert len(albums) == len(self.paths)
        if progress_reporter is not None:
            progress_reporter(
                PreparationProgress(
                    PreparationStage.RESOLVING_ARTWORK,
                    len(albums),
                    len(albums),
                    "Artwork resolved",
                )
            )
        return tuple(
            ResolvedArtwork(album, path)
            for album, path in zip(albums, self.paths, strict=True)
        )


def test_default_source_validates_but_does_not_contact_spotify() -> None:
    workflow = AlbumosaicWorkflow()

    try:
        workflow.resolve_playlist("https://open.spotify.com/playlist/abc123")
    except SpotifyPlaylistResolutionUnavailable as error:
        assert "No request was made" in str(error)
    else:
        raise AssertionError("Expected the unconfigured source to stop resolution")


def test_generate_reports_all_stages_and_frame_progress(tmp_path: Path) -> None:
    playlist = _playlist()
    artwork_paths = (tmp_path / "one.png", tmp_path / "two.png")
    Image.new("RGB", (12, 12), "red").save(artwork_paths[0])
    Image.new("RGB", (12, 12), "blue").save(artwork_paths[1])

    def fake_renderer(
        input_path: str | Path,
        album_tiles: tuple,
        tile_count: int,
        output_path: Path,
        *,
        progress_callback,
        finalizing_callback,
        blend_alpha: float,
        match_mode: MatchMode,
    ) -> Path:
        assert input_path == "input.mp4"
        assert len(album_tiles) == 2
        assert tile_count == 2
        assert blend_alpha == 0.15
        assert match_mode is MatchMode.UNIQUE_PER_FRAME
        progress_callback(0, 4)
        progress_callback(2, 4)
        progress_callback(4, 4)
        finalizing_callback()
        output_path.write_bytes(b"mp4")
        return output_path

    progress: list[WorkflowProgress] = []
    workflow = AlbumosaicWorkflow(
        playlist_source=_PlaylistSource(playlist),
        artwork_provider=_ArtworkProvider(artwork_paths),
        video_renderer=fake_renderer,
        output_dir=tmp_path / "output",
    )

    resolved = workflow.resolve_playlist("playlist-input")
    result = workflow.generate(
        resolved,
        "input.mp4",
        2,
        progress.append,
        blend_alpha=0.15,
        match_mode=MatchMode.UNIQUE_PER_FRAME,
    )

    assert result.read_bytes() == b"mp4"
    observed_stages = {item.stage for item in progress}
    assert observed_stages == set(WorkflowStage)
    render_updates = [
        item for item in progress if item.stage is WorkflowStage.RENDERING_VIDEO
    ]
    assert [(item.processed_frames, item.total_frames) for item in render_updates] == [
        (0, 4),
        (2, 4),
        (4, 4),
    ]
    assert progress[-1] == WorkflowProgress(WorkflowStage.FINISHED, 100)


def test_workflow_accepts_density_far_above_usable_album_count(tmp_path: Path) -> None:
    artwork_paths = (tmp_path / "one.png", tmp_path / "two.png")
    for path in artwork_paths:
        Image.new("RGB", (12, 12), "red").save(path)
    observed: list[tuple[int, int]] = []

    def fake_renderer(
        input_path: str | Path,
        album_tiles: tuple,
        tile_count: int,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        del input_path, kwargs
        observed.append((len(album_tiles), tile_count))
        output_path.write_bytes(b"mp4")
        return output_path

    workflow = AlbumosaicWorkflow(
        playlist_source=_PlaylistSource(_playlist()),
        artwork_provider=_ArtworkProvider(artwork_paths),
        video_renderer=fake_renderer,
        output_dir=tmp_path / "output",
    )
    prepared = workflow.prepare_artwork(_playlist())

    assert workflow.generate(prepared, "input.mp4", 1500).is_file()
    assert observed == [(2, 1500)]

    with pytest.raises(ValueError, match="between 2 and 3000"):
        workflow.generate(prepared, "input.mp4", 3001)


def test_default_source_uses_spotify_only_when_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    assert isinstance(
        default_spotify_playlist_source(), UnconfiguredSpotifyPlaylistSource
    )

    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client-id")
    assert isinstance(default_spotify_playlist_source(), SpotifyPlaylistSource)


def test_generation_requires_two_independently_resolved_albums(
    tmp_path: Path,
) -> None:
    playlist = _playlist()
    cover = tmp_path / "one.png"
    Image.new("RGB", (12, 12), "red").save(cover)

    class PartialProvider:
        def get_many(
            self,
            albums: tuple[Album, ...],
            progress_reporter: PreparationProgressReporter | None = None,
        ) -> tuple[ResolvedArtwork, ...]:
            del progress_reporter
            return (ResolvedArtwork(albums[0], cover),)

    workflow = AlbumosaicWorkflow(
        playlist_source=_PlaylistSource(playlist),
        artwork_provider=PartialProvider(),
        output_dir=tmp_path / "output",
    )

    with pytest.raises(ValueError, match="At least two albums"):
        workflow.generate(playlist, "unused.mp4", 2)


def test_exportify_uses_the_same_independent_artwork_provider(tmp_path: Path) -> None:
    csv_path = tmp_path / "playlist.csv"
    csv_path.write_text(
        "Track Name,Artist Name(s),Album Name,Album Image URL\n"
        "Song,Artist,Album,https://i.scdn.co/ignored.jpg\n",
        encoding="utf-8",
    )
    observed: list[tuple[Album, ...]] = []

    class RecordingProvider:
        def get_many(
            self,
            albums: tuple[Album, ...],
            progress_reporter: PreparationProgressReporter | None = None,
        ) -> tuple[ResolvedArtwork, ...]:
            del progress_reporter
            observed.append(albums)
            return ()

    workflow = AlbumosaicWorkflow(artwork_provider=RecordingProvider())
    with pytest.raises(ValueError, match="At least two albums"):
        workflow.prepare_exportify(csv_path)

    assert observed[0][0].album_name == "Album"


def test_playlist_preparation_reports_metadata_artwork_and_ready(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    playlist = _playlist()
    artwork_paths = (tmp_path / "one.png", tmp_path / "two.png")
    workflow = AlbumosaicWorkflow(
        playlist_source=_PlaylistSource(playlist),
        artwork_provider=_ArtworkProvider(artwork_paths),
    )
    progress: list[PreparationProgress] = []

    with caplog.at_level(logging.DEBUG, logger="app.workflow"):
        prepared = workflow.prepare_playlist("playlist-input", progress.append)

    assert prepared.usable_album_count == 2
    assert [update.stage for update in progress] == [
        PreparationStage.FETCHING_PLAYLIST,
        PreparationStage.FETCHING_PLAYLIST,
        PreparationStage.RESOLVING_ARTWORK,
        PreparationStage.READY,
    ]
    assert "2 tracks / 2 unique albums" in progress[1].message
    assert "Spotify metadata:" in caplog.text
