"""Provider-neutral progress snapshots for playlist preparation."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class PreparationStage(Enum):
    """Stages emitted while playlist metadata and artwork are prepared."""

    FETCHING_PLAYLIST = "Fetching playlist"
    RESOLVING_ARTWORK = "Resolving album artwork"
    DOWNLOADING_ARTWORK = "Downloading artwork"
    READY = "Playlist ready"


@dataclass(frozen=True, slots=True)
class PreparationProgress:
    """One UI-independent playlist preparation update."""

    stage: PreparationStage
    processed: int
    total: int
    message: str


PreparationProgressReporter = Callable[[PreparationProgress], None]
