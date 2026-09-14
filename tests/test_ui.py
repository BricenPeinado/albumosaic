"""Tests for the Gradio interface and its thin event controller."""

from pathlib import Path

import gradio as gr
import pytest

from app.mosaic.grid import GridSpec
from app.playlist.models import Album, Playlist, Track
from app.ui.app import build_interface
from app.ui.controller import AlbumosaicUIController, blend_percentage_to_alpha


def _playlist() -> Playlist:
    tracks = []
    for index in range(3):
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


class _FakeWorkflow:
    def resolve_playlist(self, playlist_url: str) -> Playlist:
        assert playlist_url == "https://open.spotify.com/playlist/abc123"
        return _playlist()

    def grid_for_video(self, video_path: str | Path, tile_count: int) -> GridSpec:
        assert str(video_path) == "video.mp4"
        assert tile_count == 3
        return GridSpec(rows=2, columns=2, tile_count=4)


def test_build_interface_returns_blocks() -> None:
    assert isinstance(build_interface(), gr.Blocks)


def test_playlist_resolution_enables_density_with_unique_album_maximum() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    updates = list(
        controller.resolve_playlist(
            "https://open.spotify.com/playlist/abc123",
            "video.mp4",
            3,
        )
    )
    final = updates[-1]

    assert final[0].unique_album_count == 3
    assert final[1] == "Found **3 tracks** across **3 unique albums**."
    assert final[2]["maximum"] == 3
    assert final[2]["interactive"] is True
    assert final[3] == "Mosaic grid: approximately **2 × 2**"
    assert final[4]["interactive"] is True


def test_generate_requires_a_resolved_playlist() -> None:
    controller = AlbumosaicUIController(_FakeWorkflow())  # type: ignore[arg-type]

    update = next(controller.generate(None, "video.mp4", 2, 0))

    assert "Resolve a playlist" in update[0]
    assert update[3] is None
    assert update[4] is None


def test_blend_percentage_is_converted_to_renderer_alpha() -> None:
    assert blend_percentage_to_alpha(0) == 0.0
    assert blend_percentage_to_alpha(15) == 0.15
    assert blend_percentage_to_alpha(50) == 0.5


@pytest.mark.parametrize("percentage", [-1, 51])
def test_blend_percentage_rejects_values_outside_ui_range(percentage: int) -> None:
    with pytest.raises(ValueError, match="between 0% and 50%"):
        blend_percentage_to_alpha(percentage)
