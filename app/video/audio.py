"""FFmpeg integration for restoring source audio."""

from pathlib import Path


def restore_audio(
    silent_video_path: Path,
    source_video_path: Path,
    output_path: Path,
) -> Path:
    """Mux the source audio into the processed video with FFmpeg."""
    raise NotImplementedError("Audio restoration is not implemented yet")
