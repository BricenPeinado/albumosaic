"""Provider-neutral application workflow for the Albumosaic UI."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from os import environ
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image

from app.mosaic.grid import GridSpec, calculate_grid
from app.mosaic.matcher import AlbumTile, MatchMode
from app.mosaic.renderer import validate_blend_alpha
from app.playlist.artwork_sources import (
    ArtworkResolver,
    IndependentArtworkProvider,
    LocalManifestArtworkResolver,
    MusicBrainzArtworkResolver,
    ResolvedArtwork,
)
from app.playlist.exportify import ExportifyCSVSource
from app.playlist.models import Album, Playlist
from app.playlist.source import PlaylistInput, PlaylistSource
from app.playlist.spotify import SpotifyPlaylistSource, parse_spotify_playlist_url
from app.playlist.spotify_auth import SpotifyOAuthConfig, SpotifyOAuthManager
from app.video.reader import VideoReader
from app.video.renderer import render_video


class WorkflowStage(Enum):
    """Ordered stages displayed while generating a mosaic video."""

    RESOLVING_PLAYLIST = (1, "Resolving playlist")
    GETTING_ARTWORK = (2, "Getting album artwork")
    ANALYZING_ARTWORK = (3, "Analyzing artwork")
    RENDERING_VIDEO = (4, "Rendering video")
    RESTORING_AUDIO = (5, "Restoring audio")
    FINISHED = (6, "Finished")

    @property
    def number(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]


@dataclass(frozen=True, slots=True)
class WorkflowProgress:
    """A UI-independent snapshot of generation progress."""

    stage: WorkflowStage
    percentage: int
    processed_frames: int = 0
    total_frames: int = 0


@dataclass(frozen=True, slots=True)
class PreparedPlaylist:
    """Playlist metadata paired with independently sourced artwork."""

    playlist: Playlist
    artwork: tuple[ResolvedArtwork, ...]

    @property
    def usable_album_count(self) -> int:
        return len(self.artwork)


class SpotifyPlaylistResolutionUnavailable(RuntimeError):
    """Raised when no live Spotify playlist provider has been configured."""


class UnconfiguredSpotifyPlaylistSource(PlaylistSource):
    """Validate Spotify input while making no Spotify network requests."""

    is_connected = False
    connection_status = "Spotify setup is missing. Set SPOTIFY_CLIENT_ID."

    def connect(self) -> None:
        raise SpotifyPlaylistResolutionUnavailable(self.connection_status)

    def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
        if not isinstance(playlist_input, str):
            raise TypeError("The Spotify playlist input must be a URL")
        parse_spotify_playlist_url(playlist_input)
        raise SpotifyPlaylistResolutionUnavailable(
            "Live Spotify playlist resolution is not configured. "
            "No request was made to Spotify."
        )


class ArtworkProvider(Protocol):
    """Small boundary used by the workflow for artwork retrieval."""

    def get_many(self, albums: tuple[Album, ...]) -> tuple[ResolvedArtwork, ...]:
        """Return independently resolved local artwork in album order."""


VideoRenderer = Callable[..., Path]
ProgressReporter = Callable[[WorkflowProgress], None]


class AlbumosaicWorkflow:
    """Coordinate playlist, artwork, mosaic, and video services."""

    def __init__(
        self,
        playlist_source: PlaylistSource | None = None,
        artwork_provider: ArtworkProvider | None = None,
        video_renderer: VideoRenderer = render_video,
        output_dir: str | Path = "output",
    ) -> None:
        self.playlist_source = playlist_source or default_spotify_playlist_source()
        self.artwork_provider = artwork_provider or default_artwork_provider()
        self.video_renderer = video_renderer
        self.output_dir = Path(output_dir)

    def resolve_playlist(self, playlist_url: str) -> Playlist:
        """Resolve a playlist through the configured provider boundary."""
        return self.playlist_source.resolve_playlist(playlist_url)

    @property
    def playlist_connection_status(self) -> str:
        return str(
            getattr(
                self.playlist_source,
                "connection_status",
                "Playlist source does not require a connection.",
            )
        )

    def connect_playlist_source(self) -> str:
        """Connect the configured interactive playlist provider."""
        connect = getattr(self.playlist_source, "connect", None)
        if not callable(connect):
            return "Playlist source does not require a connection."
        connect()
        return self.playlist_connection_status

    def prepare_playlist(self, playlist_url: str) -> PreparedPlaylist:
        """Resolve Spotify metadata and independent artwork for UI use."""
        playlist = self.resolve_playlist(playlist_url)
        return self.prepare_artwork(playlist)

    def prepare_exportify(self, csv_path: str | Path) -> PreparedPlaylist:
        """Resolve an Exportify CSV through the same independent art provider."""
        playlist = ExportifyCSVSource().resolve_playlist(csv_path)
        return self.prepare_artwork(playlist)

    def prepare_artwork(self, playlist: Playlist) -> PreparedPlaylist:
        return PreparedPlaylist(
            playlist, self.artwork_provider.get_many(playlist.albums)
        )

    def grid_for_video(
        self,
        video_path: str | Path,
        tile_count: int,
    ) -> GridSpec:
        """Calculate the visible grid from source video metadata."""
        with VideoReader(video_path) as reader:
            metadata = reader.metadata
        return calculate_grid(metadata.width, metadata.height, tile_count)

    def generate(
        self,
        playlist: Playlist | PreparedPlaylist,
        video_path: str | Path,
        tile_count: int,
        progress_reporter: ProgressReporter | None = None,
        *,
        blend_alpha: float = 0.0,
        match_mode: MatchMode = MatchMode.NEAREST,
    ) -> Path:
        """Generate a mosaic MP4 while reporting provider-neutral stages."""
        alpha = validate_blend_alpha(blend_alpha)
        if not isinstance(match_mode, MatchMode):
            raise TypeError("match_mode must be a MatchMode")
        metadata = (
            playlist.playlist if isinstance(playlist, PreparedPlaylist) else playlist
        )
        if metadata.unique_album_count < 2:
            raise ValueError("A playlist needs at least two unique albums")
        if not 2 <= tile_count <= metadata.unique_album_count:
            raise ValueError(
                "Tile count must be between 2 and the playlist's unique album count"
            )

        report = progress_reporter or (lambda _progress: None)
        report(WorkflowProgress(WorkflowStage.RESOLVING_PLAYLIST, 0))
        report(WorkflowProgress(WorkflowStage.GETTING_ARTWORK, 5))
        prepared = (
            playlist
            if isinstance(playlist, PreparedPlaylist)
            else self.prepare_artwork(playlist)
        )
        if prepared.usable_album_count < 2:
            raise ValueError(
                "At least two albums need independently resolved artwork before rendering"
            )

        report(WorkflowProgress(WorkflowStage.ANALYZING_ARTWORK, 20))
        album_tiles = _load_album_tiles(prepared.artwork)
        destination = self._new_output_path()

        def report_frames(processed: int, total: int) -> None:
            fraction = processed / total if total else 0.0
            percentage = min(90, 25 + round(fraction * 65))
            report(
                WorkflowProgress(
                    WorkflowStage.RENDERING_VIDEO,
                    percentage,
                    processed,
                    total,
                )
            )

        def report_finalizing() -> None:
            report(WorkflowProgress(WorkflowStage.RESTORING_AUDIO, 95))

        try:
            result = self.video_renderer(
                video_path,
                album_tiles,
                tile_count,
                destination,
                progress_callback=report_frames,
                finalizing_callback=report_finalizing,
                blend_alpha=alpha,
                match_mode=match_mode,
            )
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        report(WorkflowProgress(WorkflowStage.FINISHED, 100))
        return result

    def _new_output_path(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"albumosaic-{uuid4().hex[:12]}.mp4"


def _load_album_tiles(artwork: tuple[ResolvedArtwork, ...]) -> tuple[AlbumTile, ...]:
    tiles: list[AlbumTile] = []
    for item in artwork:
        with Image.open(item.path) as source:
            image = source.convert("RGB")
        tiles.append(
            AlbumTile(
                identifier="|".join(item.album.identity_key),
                image=image,
            )
        )
    return tuple(tiles)


def default_spotify_playlist_source() -> PlaylistSource:
    """Build the live source only when local Spotify configuration exists."""
    try:
        config = SpotifyOAuthConfig.from_environment()
    except RuntimeError:
        return UnconfiguredSpotifyPlaylistSource()
    return SpotifyPlaylistSource(SpotifyOAuthManager(config))


def default_artwork_provider() -> IndependentArtworkProvider:
    """Build local-first independent artwork resolution from environment."""
    resolvers: list[ArtworkResolver] = []
    manifest = environ.get("ALBUMOSAIC_ARTWORK_MANIFEST", "").strip()
    if manifest:
        resolvers.append(LocalManifestArtworkResolver(manifest))
    contact = environ.get("MUSICBRAINZ_CONTACT", "").strip()
    if contact:
        resolvers.append(MusicBrainzArtworkResolver(contact))
    return IndependentArtworkProvider(tuple(resolvers))
