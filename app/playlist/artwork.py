"""Album artwork download and cache boundary."""

from pathlib import Path
from typing import Sequence

from app.playlist.models import Album


def cache_artwork(albums: Sequence[Album], cache_dir: Path) -> tuple[Path, ...]:
    """Download and cache one cover image for each album."""
    raise NotImplementedError("Artwork caching is not implemented yet")
