"""Tests for the Gradio interface and its thin event controller."""

from pathlib import Path
from threading import Event

import gradio as gr
import pytest

from app.mosaic.grid import GridSpec
from app.mosaic.matcher import MatchMode
from app.playlist.artwork_sources import ResolvedArtwork
from app.playlist.models import Album, Playlist, Track
from app.playlist.progress import (
    PreparationProgress,
    PreparationProgressReporter,
    PreparationStage,
)
from app.ui.app import build_interface
from app.ui.controller import (
    AlbumosaicUIController,
    blend_percentage_to_alpha,
    match_mode_from_unique,
)
from app.workflow import PreparedPlaylist


def _playlist() -> Playlist:
    tracks = []
    for index in range(3):
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


class _FakeWorkflow:
    last_match_mode: MatchMode | None = None

    def prepare_playlist(
        self,
        playlist_url: str,
        progress_reporter: PreparationProgressReporter | None = None,
    ) -> PreparedPlaylist:
        assert playlist_url == "https://open.spotify.com/playlist/abc123"
        playlist = _playlist()
        if progress_reporter is not None:
            progress_reporter(
                PreparationProgress(
                    PreparationStage.RESOLVING_ARTWORK,
                    1,
                    3,
                    'Finding artwork for album 1 / 3\n"Album 0" — Artist',
                )
            )
        artwork = tuple(
            ResolvedArtwork(album, Path(f"{index}.png"))
            for index, album in enumerate(playlist.albums)
        )
        return PreparedPlaylist(playlist, artwork)

    def grid_for_video(self, video_path: str | Path, tile_count: int) -> GridSpec:
        assert str(video_path) == "video.mp4"
        assert tile_count == 3
        return GridSpec(rows=2, columns=2, tile_count=4)

    def generate(self, *args: object, match_mode: MatchMode, **kwargs: object) -> Path:
        del args, kwargs
        self.last_match_mode = match_mode
        return Path("result.mp4")


def test_build_interface_returns_blocks() -> None:
    assert isinstance(build_interface(), gr.Blocks)


def test_playlist_resolution_enables_density_with_unique_album_maximum() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    updates = list(
        controller.resolve_playlist(
            "https://open.spotify.com/playlist/abc123",
            "video.mp4",
            3,
            False,
        )
    )
    final = updates[-1]

    assert final[0].playlist.unique_album_count == 3
    assert final[1] == (
        "Found **3 tracks** across **3 unique albums**.  \n"
        "Independent artwork found for **3 albums**."
    )
    assert final[2]["maximum"] == 3
    assert final[2]["interactive"] is True
    assert final[3] == "Mosaic grid: approximately **2 × 2** (4 tiles)"
    assert final[4]["interactive"] is True


def test_controller_streams_preparation_progress_before_completion() -> None:
    release = Event()

    class BlockingWorkflow(_FakeWorkflow):
        def prepare_playlist(
            self,
            playlist_url: str,
            progress_reporter: PreparationProgressReporter | None = None,
        ) -> PreparedPlaylist:
            assert progress_reporter is not None
            progress_reporter(
                PreparationProgress(
                    PreparationStage.RESOLVING_ARTWORK,
                    4,
                    18,
                    'Resolving album artwork 4 / 18\n"Loveless" — My Bloody Valentine',
                )
            )
            assert release.wait(timeout=1)
            return super().prepare_playlist(playlist_url)

    controller = AlbumosaicUIController(BlockingWorkflow())  # type: ignore[arg-type]
    updates = controller.resolve_playlist(
        "https://open.spotify.com/playlist/abc123",
        "video.mp4",
        3,
        False,
    )

    initial = next(updates)
    streamed = next(updates)

    assert "Fetching playlist" in initial[5]
    assert "Resolving album artwork 4 / 18" in streamed[1]
    assert streamed[6] > 0
    assert streamed[0] is None

    release.set()
    final = list(updates)[-1]
    assert isinstance(final[0], PreparedPlaylist)


def test_generate_requires_a_resolved_playlist() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    update = next(controller.generate(None, "video.mp4", 2, 0, False))

    assert "Resolve a playlist" in update[0]
    assert update[3] is None
    assert update[4] is None


def test_blend_percentage_is_converted_to_renderer_alpha() -> None:
    assert blend_percentage_to_alpha(0) == 0.0
    assert blend_percentage_to_alpha(15) == 0.15
    assert blend_percentage_to_alpha(50) == 0.5


def test_unique_checkbox_is_passed_to_workflow_as_match_mode() -> None:
    workflow = _FakeWorkflow()
    controller = AlbumosaicUIController(workflow)  # type: ignore[arg-type]

    updates = list(controller.generate(_playlist(), "video.mp4", 3, 0, True))

    assert updates[-1][3] == "result.mp4"
    assert workflow.last_match_mode is MatchMode.UNIQUE_PER_FRAME


def test_grid_summary_explains_unavoidable_repeats() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    summary = controller.grid_text("video.mp4", 3, _playlist(), True)

    assert "4 tiles" in summary
    assert "up to 1 repeat may be required" in summary


def test_checkbox_conversion_defaults_to_nearest() -> None:
    assert match_mode_from_unique(False) is MatchMode.NEAREST
    assert match_mode_from_unique(True) is MatchMode.UNIQUE_PER_FRAME


@pytest.mark.parametrize("percentage", [-1, 51])
def test_blend_percentage_rejects_values_outside_ui_range(percentage: int) -> None:
    with pytest.raises(ValueError, match="between 0% and 50%"):
        blend_percentage_to_alpha(percentage)
