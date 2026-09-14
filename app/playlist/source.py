"""Abstract boundary for local or future remote playlist providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TextIO, TypeAlias

from app.playlist.models import Playlist

PlaylistInput: TypeAlias = str | Path | TextIO


class PlaylistSource(ABC):
    """Resolve provider-specific input into the common playlist domain."""

    @abstractmethod
    def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
        """Resolve source input without exposing provider details downstream."""
