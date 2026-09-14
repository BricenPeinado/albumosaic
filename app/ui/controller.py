"""Gradio event adapters for the provider-neutral Albumosaic workflow."""

from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import gradio as gr

from app.playlist.models import Playlist
from app.workflow import AlbumosaicWorkflow, WorkflowProgress


class AlbumosaicUIController:
    """Translate workflow results into small Gradio event responses."""

    def __init__(self, workflow: AlbumosaicWorkflow) -> None:
        self.workflow = workflow

    def resolve_playlist(
        self,
        playlist_url: str,
        video_path: str | None,
        tile_count: int,
    ) -> Iterator[tuple[Any, ...]]:
        """Resolve playlist input and enable density-dependent controls."""
        yield (
            None,
            "Resolving playlist…",
            gr.update(minimum=2, maximum=3, value=2, interactive=False),
            "Mosaic grid: add a video after resolving your playlist.",
            gr.update(interactive=False),
            self._stage_text(1, "Resolving playlist"),
            0,
            "Frames processed: —",
        )

        try:
            playlist = self.workflow.resolve_playlist(playlist_url)
            album_count = playlist.unique_album_count
            if album_count < 2:
                raise ValueError("The playlist must contain at least 2 unique albums")
            selected_count = min(max(2, tile_count), album_count)
            yield (
                playlist,
                (
                    f"Found **{len(playlist.tracks)} tracks** across "
                    f"**{album_count} unique albums**."
                ),
                gr.update(
                    minimum=2,
                    maximum=album_count,
                    value=selected_count,
                    interactive=True,
                ),
                self.grid_text(video_path, selected_count),
                gr.update(interactive=bool(video_path)),
                "Playlist ready",
                0,
                "Frames processed: —",
            )
        except Exception as error:
            yield (
                None,
                f"Could not resolve playlist: {error}",
                gr.update(minimum=2, maximum=3, value=2, interactive=False),
                "Mosaic grid: unavailable",
                gr.update(interactive=False),
                "Playlist resolution stopped",
                0,
                "Frames processed: —",
            )

    def update_readiness(
        self,
        video_path: str | None,
        tile_count: int,
        playlist: Playlist | None,
    ) -> tuple[str, dict[str, Any]]:
        """Refresh the grid estimate and Generate button state."""
        ready = playlist is not None and bool(video_path)
        return self.grid_text(video_path, tile_count), gr.update(interactive=ready)

    def grid_text(self, video_path: str | None, tile_count: int) -> str:
        """Return a concise aspect-aware grid summary."""
        if not video_path:
            return "Mosaic grid: add a video to see the estimate."
        try:
            grid = self.workflow.grid_for_video(video_path, tile_count)
        except Exception as error:
            return f"Mosaic grid: unavailable ({error})"
        return f"Mosaic grid: approximately **{grid.columns} × {grid.rows}**"

    def generate(
        self,
        playlist: Playlist | None,
        video_path: str | None,
        tile_count: int,
        blend_percentage: float,
    ) -> Iterator[tuple[Any, ...]]:
        """Run the blocking workflow in a worker and stream its progress."""
        if playlist is None:
            yield self._error_result("Resolve a playlist before generating.")
            return
        if not video_path:
            yield self._error_result("Add a source video before generating.")
            return

        updates: Queue[tuple[str, object]] = Queue()

        def run_workflow() -> None:
            try:
                result = self.workflow.generate(
                    playlist,
                    video_path,
                    tile_count,
                    progress_reporter=lambda progress: updates.put(
                        ("progress", progress)
                    ),
                    blend_alpha=blend_percentage_to_alpha(blend_percentage),
                )
            except Exception as error:
                updates.put(("error", error))
            else:
                updates.put(("result", result))

        Thread(target=run_workflow, daemon=True).start()
        yield (
            self._stage_text(1, "Resolving playlist"),
            0,
            "Frames processed: 0 / —",
            None,
            None,
            gr.update(interactive=False),
        )

        while True:
            kind, payload = updates.get()
            if kind == "progress":
                assert isinstance(payload, WorkflowProgress)
                yield (
                    self._stage_text(payload.stage.number, payload.stage.label),
                    payload.percentage,
                    self._frames_text(payload),
                    None,
                    None,
                    gr.update(interactive=False),
                )
                continue
            if kind == "error":
                assert isinstance(payload, Exception)
                yield self._error_result(str(payload))
                return

            if not isinstance(payload, Path):
                yield self._error_result("Generation returned an invalid output path.")
                return
            result = payload
            yield (
                self._stage_text(6, "Finished"),
                100,
                "Rendering complete",
                str(result),
                str(result),
                gr.update(interactive=True),
            )
            return

    @staticmethod
    def _stage_text(number: int, label: str) -> str:
        return f"### Stage {number} of 6 · {label}"

    @staticmethod
    def _frames_text(progress: WorkflowProgress) -> str:
        if progress.total_frames:
            return (
                f"Frames processed: **{progress.processed_frames:,} / "
                f"{progress.total_frames:,}** · **{progress.percentage}%**"
            )
        return f"**{progress.percentage}%**"

    @classmethod
    def _error_result(cls, message: str) -> tuple[Any, ...]:
        return (
            f"### Generation stopped\n{message}",
            0,
            "Frames processed: —",
            None,
            None,
            gr.update(interactive=True),
        )


def blend_percentage_to_alpha(blend_percentage: float) -> float:
    """Convert the UI's percentage value into a renderer alpha."""
    value = float(blend_percentage)
    if not 0.0 <= value <= 50.0:
        raise ValueError("Original video blend must be between 0% and 50%")
    return value / 100.0


def format_blend_percentage(blend_percentage: float) -> str:
    """Format the integer UI slider value as a percentage."""
    return f"Selected blend: **{round(blend_percentage)}%**"
