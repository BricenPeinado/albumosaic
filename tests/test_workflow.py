"""Tests for provider-neutral UI workflow orchestration."""

from pathlib import Path

from PIL import Image

from app.mosaic.matcher import MatchMode
from app.playlist.models import Album, Playlist, Track
from app.playlist.source import PlaylistInput, PlaylistSource
from app.workflow import (
    AlbumosaicWorkflow,
    SpotifyPlaylistResolutionUnavailable,
    WorkflowProgress,
    WorkflowStage,
)


def _playlist() -> Playlist:
    tracks = []
    for index in range(2):
        album = Album(
            album_id=f"album-{index}",
            album_name=f"Album {index}",
            artists=("Artist",),
            artwork_url=f"https://example.test/{index}.png",
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

    def get_many(self, albums: tuple[Album, ...]) -> tuple[Path, ...]:
        assert len(albums) == len(self.paths)
        return self.paths


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
