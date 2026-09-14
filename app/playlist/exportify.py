"""Local-only playlist ingestion for Exportify CSV files."""

from collections.abc import Mapping, Sequence
from csv import DictReader
from pathlib import Path
from typing import TextIO
from urllib.parse import urlparse

from app.playlist.models import Album, Playlist, Track
from app.playlist.source import PlaylistInput, PlaylistSource

_COLUMN_ALIASES = {
    "track_uri": ("Track URI", "Spotify ID", "Track URL"),
    "track_name": ("Track Name", "Title"),
    "track_artists": ("Artist Name(s)", "Artists"),
    "album_uri": ("Album URI", "Album URL"),
    "album_name": ("Album Name", "Album"),
    "album_artists": ("Album Artist Name(s)", "Album Artists"),
    "artwork_url": ("Album Image URL", "Artwork URL"),
    "release_date": ("Album Release Date", "Release Date"),
    "disc_number": ("Disc Number",),
    "track_number": ("Track Number",),
    "duration_ms": ("Track Duration (ms)", "Duration_ms"),
    "preview_url": ("Track Preview URL", "Preview URL"),
    "explicit": ("Explicit?", "Explicit"),
    "popularity": ("Popularity",),
    "isrc": ("ISRC", "Track ISRC"),
    "added_by": ("Added By",),
    "added_at": ("Added At",),
}
_REQUIRED_COLUMNS = ("track_name", "track_artists", "album_name")
CSVValue = str | list[str] | None
CSVRow = Mapping[str | None, CSVValue]


class ExportifyCSVError(ValueError):
    """Raised when an Exportify CSV is empty or structurally invalid."""


class ExportifyCSVSource(PlaylistSource):
    """Resolve an Exportify UTF-8 CSV without making network requests."""

    def __init__(
        self,
        *,
        playlist_name: str | None = None,
        playlist_id: str | None = None,
        source_url: str | None = None,
    ) -> None:
        self.playlist_name = playlist_name
        self.playlist_id = playlist_id
        self.source_url = source_url

    def resolve_playlist(self, playlist_input: PlaylistInput) -> Playlist:
        """Parse tracks and return a playlist with deduplicated albums."""
        if isinstance(playlist_input, (str, Path)):
            path = Path(playlist_input)
            if not path.is_file():
                raise FileNotFoundError(f"Exportify CSV does not exist: {path}")
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                return self._parse(stream, default_name=path.stem)

        default_name = _stream_name(playlist_input)
        return self._parse(playlist_input, default_name=default_name)

    def _parse(self, stream: TextIO, *, default_name: str) -> Playlist:
        reader = DictReader(stream)
        if reader.fieldnames is None:
            raise ExportifyCSVError("Exportify CSV is empty or has no header row")
        columns = _resolve_columns(reader.fieldnames)

        tracks: list[Track] = []
        for row_number, row in enumerate(reader, start=2):
            if not any(
                isinstance(value, str) and value.strip() for value in row.values()
            ):
                continue
            tracks.append(_parse_track(row, columns, row_number))

        return Playlist(
            playlist_id=self.playlist_id,
            playlist_name=self.playlist_name or default_name,
            source_url=self.source_url,
            tracks=tuple(tracks),
        )


def _resolve_columns(fieldnames: Sequence[str | None]) -> dict[str, str]:
    normalized_headers = {
        _normalize_header(fieldname): fieldname
        for fieldname in fieldnames
        if fieldname is not None
    }
    columns: dict[str, str] = {}
    for field, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            header = normalized_headers.get(_normalize_header(alias))
            if header is not None:
                columns[field] = header
                break

    missing = [field for field in _REQUIRED_COLUMNS if field not in columns]
    if missing:
        readable = ", ".join(missing)
        raise ExportifyCSVError(
            f"Exportify CSV is missing required columns: {readable}"
        )
    return columns


def _parse_track(
    row: CSVRow,
    columns: dict[str, str],
    row_number: int,
) -> Track:
    track_name = _required_value(row, columns, "track_name", row_number)
    album_name = _required_value(row, columns, "album_name", row_number)
    track_artists = _split_artists(
        _required_value(row, columns, "track_artists", row_number)
    )
    album_artist_value = _value(row, columns, "album_artists")
    album_artists = (
        _split_artists(album_artist_value) if album_artist_value else track_artists
    )
    track_uri = _value(row, columns, "track_uri")
    album_uri = _value(row, columns, "album_uri")

    parent_album = Album(
        album_id=_source_id(album_uri, "album"),
        album_name=album_name,
        artists=album_artists,
        artwork_url=_value(row, columns, "artwork_url"),
        source_url=_source_url(album_uri, "album"),
        release_date=_value(row, columns, "release_date"),
    )
    return Track(
        track_id=_source_id(track_uri, "track"),
        track_name=track_name,
        artists=track_artists,
        album=parent_album,
        source_url=_source_url(track_uri, "track"),
        disc_number=_optional_int(row, columns, "disc_number", row_number),
        track_number=_optional_int(row, columns, "track_number", row_number),
        duration_ms=_optional_int(row, columns, "duration_ms", row_number),
        preview_url=_value(row, columns, "preview_url"),
        explicit=_optional_bool(row, columns, "explicit", row_number),
        popularity=_optional_int(row, columns, "popularity", row_number),
        isrc=_value(row, columns, "isrc"),
        added_by=_value(row, columns, "added_by"),
        added_at=_value(row, columns, "added_at"),
    )


def _value(row: CSVRow, columns: dict[str, str], field: str) -> str | None:
    header = columns.get(field)
    if header is None:
        return None
    value = row.get(header)
    stripped = value.strip() if isinstance(value, str) else ""
    return stripped or None


def _required_value(
    row: CSVRow,
    columns: dict[str, str],
    field: str,
    row_number: int,
) -> str:
    value = _value(row, columns, field)
    if value is None:
        raise ExportifyCSVError(f"Missing {field} in CSV row {row_number}")
    return value


def _optional_int(
    row: CSVRow,
    columns: dict[str, str],
    field: str,
    row_number: int,
) -> int | None:
    value = _value(row, columns, field)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ExportifyCSVError(
            f"Invalid integer for {field} in CSV row {row_number}: {value}"
        ) from error


def _optional_bool(
    row: CSVRow,
    columns: dict[str, str],
    field: str,
    row_number: int,
) -> bool | None:
    value = _value(row, columns, field)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ExportifyCSVError(
        f"Invalid boolean for {field} in CSV row {row_number}: {value}"
    )


def _split_artists(value: str) -> tuple[str, ...]:
    separator = ";" if ";" in value else ","
    artists = tuple(part.strip() for part in value.split(separator) if part.strip())
    return artists or (value.strip(),)


def _source_id(value: str | None, entity: str) -> str | None:
    if value is None:
        return None
    prefix = f"spotify:{entity}:"
    if value.startswith(prefix):
        return value.removeprefix(prefix) or None

    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc.endswith("spotify.com") and entity in path_parts:
        entity_index = path_parts.index(entity)
        if entity_index + 1 < len(path_parts):
            return path_parts[entity_index + 1]
    return value if ":" not in value and "/" not in value else None


def _source_url(value: str | None, entity: str) -> str | None:
    if value is None:
        return None
    if value.startswith(("https://", "http://")):
        return value
    source_id = _source_id(value, entity)
    if source_id is None:
        return None
    return f"https://open.spotify.com/{entity}/{source_id}"


def _normalize_header(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _stream_name(stream: TextIO) -> str:
    name = getattr(stream, "name", None)
    if isinstance(name, str) and name:
        return Path(name).stem
    return "Exportify Playlist"
