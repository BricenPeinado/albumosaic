"""Tests for the Gradio interface and its thin event controller."""

from pathlib import Path
from threading import Event

import gradio as gr
import pytest

from app.mosaic.grid import (
    GridSpec,
    calculate_grid,
    maximum_render_density,
    recommended_density_range,
)
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
        return calculate_grid(1920, 1080, tile_count)

    def density_limit_for_video(self, video_path: str | Path) -> int:
        assert str(video_path) == "video.mp4"
        return maximum_render_density(1920, 1080)

    def recommended_density_for_video(self, video_path: str | Path) -> tuple[int, int]:
        assert str(video_path) == "video.mp4"
        return recommended_density_range(1920, 1080)

    def generate(self, *args: object, match_mode: MatchMode, **kwargs: object) -> Path:
        del args, kwargs
        self.last_match_mode = match_mode
        return Path("result.mp4")


def test_build_interface_returns_blocks() -> None:
    assert isinstance(build_interface(), gr.Blocks)


def test_density_slider_has_album_independent_default_and_range() -> None:
    config = build_interface().get_config_file()
    density = next(
        component["props"]
        for component in config["components"]
        if component["props"].get("label") == "Mosaic density"
    )

    assert (density["minimum"], density["maximum"], density["value"]) == (
        25,
        3000,
        200,
    )


def test_playlist_resolution_keeps_density_independent_of_album_count() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    updates = list(
        controller.resolve_playlist(
            "https://open.spotify.com/playlist/abc123",
            "video.mp4",
            200,
            False,
        )
    )
    final = updates[-1]

    assert final[0].playlist.unique_album_count == 3
    assert final[1] == (
        "Found **3 tracks** across **3 unique albums**.  \n"
        "Independent artwork found for **3 albums**.  \n"
        "Album library: **3 covers**. Mosaic density: **200 requested tiles**."
    )
    assert final[2]["minimum"] == 25
    assert final[2]["maximum"] == 3000
    assert final[2]["value"] == 200
    assert final[2]["interactive"] is True
    assert "Album library: **3 covers**" in final[3]
    assert "Mosaic density: **200 requested tiles**" in final[3]
    assert "Mosaic grid: **18 × 11 = 198 actual tiles**" in final[3]
    assert final[4]["interactive"] is True


def test_video_upload_preserves_full_slider_range_and_selected_density() -> None:
    class SmallVideoWorkflow(_FakeWorkflow):
        def grid_for_video(self, video_path: str | Path, tile_count: int) -> GridSpec:
            assert str(video_path) == "small.mp4"
            return calculate_grid(320, 180, tile_count)

        def recommended_density_for_video(
            self, video_path: str | Path
        ) -> tuple[int, int]:
            assert str(video_path) == "small.mp4"
            return recommended_density_range(320, 180)

    controller = AlbumosaicUIController(SmallVideoWorkflow())  # type: ignore[arg-type]
    summary, button, slider = controller.update_video_readiness(
        "small.mp4", 1500, _playlist(), False
    )

    assert slider["maximum"] == 3000
    assert slider["value"] == 1500
    assert "52 × 29 = 1508 actual tiles" in summary
    assert "Recommended for this video" in summary
    assert "produce very small tiles" in summary
    assert button["interactive"] is True


def test_high_density_status_warns_about_render_time() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    summary = controller.grid_text("video.mp4", 1500, _playlist())

    assert "Mosaic density: **1500 requested tiles**" in summary
    assert "actual tiles" in summary
    assert "may significantly increase render time" in summary


def test_low_resolution_warning_does_not_block_generation() -> None:
    class SmallVideoWorkflow(_FakeWorkflow):
        def grid_for_video(self, video_path: str | Path, tile_count: int) -> GridSpec:
            assert str(video_path) == "small.mp4"
            return calculate_grid(640, 480, tile_count)

        def recommended_density_for_video(
            self, video_path: str | Path
        ) -> tuple[int, int]:
            assert str(video_path) == "small.mp4"
            return recommended_density_range(640, 480)

    controller = AlbumosaicUIController(SmallVideoWorkflow())  # type: ignore[arg-type]

    summary, button = controller.update_readiness("small.mp4", 2000, _playlist(), False)

    assert "2000 requested tiles" in summary
    assert "actual tiles" in summary
    assert "100-500 tiles" in summary
    assert "produce very small tiles" in summary
    assert button["interactive"] is True


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

    summary = controller.grid_text("video.mp4", 200, _playlist(), True)

    assert "actual tiles" in summary
    assert "up to 195 repeats may be required" in summary


def test_video_dimension_limit_disables_generate_until_density_is_lowered() -> None:
    class TinyVideoWorkflow(_FakeWorkflow):
        def grid_for_video(self, video_path: str | Path, tile_count: int) -> GridSpec:
            del video_path, tile_count
            raise ValueError("cells must be at least 8 pixels wide and tall")

    controller = AlbumosaicUIController(TinyVideoWorkflow())  # type: ignore[arg-type]

    summary, button = controller.update_readiness("video.mp4", 200, _playlist(), False)

    assert "unavailable" in summary
    assert button["interactive"] is False


def test_checkbox_conversion_defaults_to_nearest() -> None:
    assert match_mode_from_unique(False) is MatchMode.NEAREST
    assert match_mode_from_unique(True) is MatchMode.UNIQUE_PER_FRAME


@pytest.mark.parametrize("percentage", [-1, 51])
def test_blend_percentage_rejects_values_outside_ui_range(percentage: int) -> None:
    with pytest.raises(ValueError, match="between 0% and 50%"):
        blend_percentage_to_alpha(percentage)
