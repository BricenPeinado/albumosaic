"""Independent local and MusicBrainz artwork resolution strategies."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from json import dumps, loads
from pathlib import Path
from re import sub
from threading import Lock
from time import monotonic, perf_counter, sleep
from typing import Protocol, cast
from unicodedata import normalize as unicode_normalize
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request

from app.network import open_url
from app.playlist.artwork import ArtworkCache, ArtworkCacheError
from app.playlist.models import Album, AlbumKey, deduplicate_albums
from app.playlist.progress import (
    PreparationProgress,
    PreparationProgressReporter,
    PreparationStage,
)

_MUSICBRAINZ_HOST = "musicbrainz.org"
_COVER_ART_HOST = "coverartarchive.org"
_ARCHIVE_HOSTS = frozenset({_COVER_ART_HOST, "archive.org"})
_DEFAULT_RESOLUTION_WORKERS = 4

logger = logging.getLogger(__name__)


class ArtworkResolutionError(RuntimeError):
    """Raised when an artwork service returns an invalid or unsafe response."""


@dataclass(frozen=True, slots=True)
class ArtworkReference:
    """An independently sourced local path or remote artwork URL."""

    provider: str
    identifier: str
    url: str | None = None
    local_path: Path | None = None

    def __post_init__(self) -> None:
        if bool(self.url) == bool(self.local_path):
            raise ValueError("ArtworkReference requires exactly one URL or local path")


@dataclass(frozen=True, slots=True)
class ResolvedArtwork:
    """An album paired with one validated local image."""

    album: Album
    path: Path


class ArtworkResolver(Protocol):
    """Resolve album identity into independently sourced artwork."""

    def resolve(self, album: Album) -> ArtworkReference | None:
        """Return a confident artwork reference or None."""


class LocalManifestArtworkResolver:
    """Resolve user-owned images from an explicit JSON manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Artwork manifest does not exist: {manifest_path}")
        raw = loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ArtworkResolutionError("Artwork manifest must contain a JSON list")
        self._entries: dict[tuple[str, str], Path] = {}
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ArtworkResolutionError(
                    f"Invalid artwork manifest item {index + 1}"
                )
            artist = entry.get("artist")
            album = entry.get("album")
            path = entry.get("path")
            if not all(
                isinstance(value, str) and value.strip()
                for value in (artist, album, path)
            ):
                raise ArtworkResolutionError(
                    f"Invalid artwork manifest item {index + 1}"
                )
            candidate = Path(cast(str, path))
            if not candidate.is_absolute():
                candidate = self.manifest_path.parent / candidate
            self._entries[
                (_normalize(cast(str, artist)), _normalize(cast(str, album)))
            ] = candidate.resolve()

    def resolve(self, album: Album) -> ArtworkReference | None:
        artist = " ".join(album.artists)
        path = self._entries.get((_normalize(artist), _normalize(album.album_name)))
        if path is None:
            return None
        if not path.is_file():
            raise ArtworkResolutionError(f"Local artwork does not exist: {path}")
        return ArtworkReference(
            provider="local-manifest",
            identifier=str(path),
            local_path=path,
        )


class MusicBrainzArtworkResolver:
    """Resolve confident release-group matches through MusicBrainz and CAA."""

    def __init__(
        self,
        contact: str,
        cache_path: str | Path = "cache/metadata/musicbrainz-artwork.json",
        *,
        search_timeout: float = 7.0,
        cover_art_timeout: float = 5.0,
        minimum_interval: float = 1.0,
        confidence_threshold: float = 75.0,
    ) -> None:
        if not contact.strip():
            raise ValueError("MusicBrainz contact information is required")
        if search_timeout <= 0 or cover_art_timeout <= 0 or minimum_interval < 0:
            raise ValueError("MusicBrainz timing values are invalid")
        self.user_agent = f"Albumosaic/0.1 ({contact.strip()})"
        self.cache_path = Path(cache_path)
        self.search_timeout = search_timeout
        self.cover_art_timeout = cover_art_timeout
        self.minimum_interval = minimum_interval
        self.confidence_threshold = confidence_threshold
        self._lock = Lock()
        self._mapping_lock = Lock()
        self._last_request = 0.0
        self._mapping = self._load_mapping()

    def resolve(self, album: Album) -> ArtworkReference | None:
        label = _album_label(album)
        key = _mapping_key(album.identity_key)
        with self._mapping_lock:
            mapping_cached = key in self._mapping
            cached_mbid = self._mapping.get(key)
        if mapping_cached:
            logger.debug("%s\nMusicBrainz mapping cache: HIT", label)
            if cached_mbid is None:
                logger.debug("unresolved")
                return None
            logger.debug("CAA lookup: cached")
            logger.debug("resolved")
            return _cover_art_reference(cached_mbid)

        logger.debug("%s\nMusicBrainz mapping cache: MISS", label)

        artist = " ".join(album.artists)
        query = (
            f'releasegroup:"{_lucene(album.album_name)}" AND artist:"{_lucene(artist)}"'
        )
        url = f"https://{_MUSICBRAINZ_HOST}/ws/2/release-group/?" + urlencode(
            {"query": query, "fmt": "json", "limit": 10}
        )
        search_started = perf_counter()
        try:
            payload = self._get_json(url)
        except ArtworkResolutionError as error:
            logger.debug(
                "MusicBrainz: %s after %.2fs",
                error,
                perf_counter() - search_started,
            )
            logger.debug("skipped")
            raise
        logger.debug("MusicBrainz: %.2fs", perf_counter() - search_started)
        candidates = payload.get("release-groups")
        if not isinstance(candidates, list):
            raise ArtworkResolutionError("MusicBrainz returned invalid search results")
        ranked = sorted(
            (
                (_candidate_score(album, candidate), candidate)
                for candidate in candidates
                if isinstance(candidate, dict)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.confidence_threshold:
            self._remember(key, None)
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 5:
            self._remember(key, None)
            return None
        mbid = ranked[0][1].get("id")
        if not isinstance(mbid, str) or not mbid:
            self._remember(key, None)
            return None
        cover_started = perf_counter()
        try:
            reference = self._cover_reference(mbid)
        except ArtworkResolutionError as error:
            logger.debug(
                "CAA lookup: %s after %.2fs",
                error,
                perf_counter() - cover_started,
            )
            logger.debug("skipped")
            raise
        logger.debug("CAA lookup: %.2fs", perf_counter() - cover_started)
        self._remember(key, mbid if reference is not None else None)
        logger.debug("resolved" if reference is not None else "unresolved")
        return reference

    def _cover_reference(self, mbid: str) -> ArtworkReference | None:
        url = f"https://{_COVER_ART_HOST}/release-group/{quote(mbid)}"
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
        )
        try:
            with open_url(request, timeout=self.cover_art_timeout) as response:
                final_url = urlparse(response.geturl())
                final_host = (final_url.hostname or "").lower()
                if final_url.scheme != "https" or not (
                    final_host == _COVER_ART_HOST
                    or final_host == "archive.org"
                    or final_host.endswith(".archive.org")
                ):
                    raise ArtworkResolutionError(
                        "Refused an unexpected Cover Art Archive redirect"
                    )
                payload = loads(response.read())
        except HTTPError as error:
            if error.code == 404:
                return None
            raise ArtworkResolutionError(
                f"Cover Art Archive returned HTTP {error.code}"
            ) from error
        except TimeoutError as error:
            raise ArtworkResolutionError(
                f"Cover Art Archive timed out after {self.cover_art_timeout:g}s"
            ) from error
        except (URLError, OSError, ValueError) as error:
            raise ArtworkResolutionError("Cover Art Archive request failed") from error
        if not isinstance(payload, dict):
            raise ArtworkResolutionError("Cover Art Archive returned invalid JSON")
        images = payload.get("images")
        if not isinstance(images, list):
            raise ArtworkResolutionError("Cover Art Archive returned invalid JSON")
        for image in images:
            if not isinstance(image, dict) or image.get("front") is not True:
                continue
            thumbnails = image.get("thumbnails")
            thumbnail = thumbnails.get("500") if isinstance(thumbnails, dict) else None
            image_url = thumbnail or image.get("image")
            if isinstance(image_url, str):
                return _cover_art_reference(mbid, image_url)
        return None

    def _get_json(self, url: str) -> dict[str, object]:
        if urlparse(url).hostname != _MUSICBRAINZ_HOST:
            raise ArtworkResolutionError("Refused an unexpected MusicBrainz URL")
        with self._lock:
            delay = self.minimum_interval - (monotonic() - self._last_request)
            if delay > 0:
                sleep(delay)
            self._last_request = monotonic()
            request = Request(url, headers={"User-Agent": self.user_agent})
            try:
                with open_url(request, timeout=self.search_timeout) as response:
                    decoded = loads(response.read())
            except HTTPError as error:
                raise ArtworkResolutionError(
                    f"MusicBrainz returned HTTP {error.code}"
                ) from error
            except TimeoutError as error:
                raise ArtworkResolutionError(
                    f"MusicBrainz timed out after {self.search_timeout:g}s"
                ) from error
            except (URLError, OSError, ValueError) as error:
                raise ArtworkResolutionError("MusicBrainz request failed") from error
        if not isinstance(decoded, dict):
            raise ArtworkResolutionError("MusicBrainz returned invalid JSON")
        return cast(dict[str, object], decoded)

    def _load_mapping(self) -> dict[str, str | None]:
        if not self.cache_path.is_file():
            return {}
        try:
            decoded = loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(decoded, dict):
            return {}
        return {
            str(key): value
            for key, value in decoded.items()
            if isinstance(value, str) and value
        }

    def _remember(self, key: str, mbid: str | None) -> None:
        with self._mapping_lock:
            if mbid is None:
                self._mapping.pop(key, None)
            else:
                self._mapping[key] = mbid
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            try:
                temporary.write_text(dumps(self._mapping, indent=2), encoding="utf-8")
                temporary.replace(self.cache_path)
            finally:
                temporary.unlink(missing_ok=True)


class IndependentArtworkProvider:
    """Apply ordered resolvers and cache only independent/local artwork."""

    def __init__(
        self,
        resolvers: tuple[ArtworkResolver, ...],
        cache: ArtworkCache | None = None,
    ) -> None:
        self.resolvers = resolvers
        self.cache = cache or ArtworkCache(
            allowed_hosts={"coverartarchive.org", "archive.org"}
        )

    def get_many(
        self,
        albums: tuple[Album, ...],
        progress_reporter: PreparationProgressReporter | None = None,
    ) -> tuple[ResolvedArtwork, ...]:
        report = progress_reporter or (lambda _progress: None)
        unique_albums = deduplicate_albums(albums)
        total = len(unique_albums)
        resolution_started = perf_counter()
        selected_by_index: dict[int, tuple[Album, ArtworkReference]] = {}
        with ThreadPoolExecutor(
            max_workers=min(_DEFAULT_RESOLUTION_WORKERS, total or 1)
        ) as executor:
            resolution_futures = {
                executor.submit(self._resolve_one, album, index, total, report): (
                    index,
                    album,
                )
                for index, album in enumerate(unique_albums)
            }
            for completed, resolution_future in enumerate(
                as_completed(resolution_futures), start=1
            ):
                index, album = resolution_futures[resolution_future]
                reference = resolution_future.result()
                if reference is not None:
                    selected_by_index[index] = (album, reference)
                report(
                    PreparationProgress(
                        PreparationStage.RESOLVING_ARTWORK,
                        completed,
                        total,
                        (
                            f"Resolving album artwork {completed} / {total}\n"
                            f"{_album_label(album)}\n"
                            f"{'resolved' if reference is not None else 'skipped'}"
                        ),
                    )
                )
        resolution_elapsed = perf_counter() - resolution_started
        selected = [selected_by_index[index] for index in sorted(selected_by_index)]

        download_started = perf_counter()
        results_by_index: dict[int, ResolvedArtwork] = {}
        cache_hits = 0
        cache_misses = 0
        with ThreadPoolExecutor(max_workers=min(4, len(selected) or 1)) as executor:
            cache_futures = {
                executor.submit(self._cache_one, item): (index, item[0])
                for index, item in enumerate(selected)
            }
            for completed, cache_future in enumerate(
                as_completed(cache_futures), start=1
            ):
                index, album = cache_futures[cache_future]
                result, cache_hit, elapsed = cache_future.result()
                cache_hits += int(cache_hit)
                cache_misses += int(not cache_hit)
                if result is not None:
                    results_by_index[index] = result
                logger.debug(
                    "%s\nartwork cache: %s, %.2fs\n%s",
                    _album_label(album),
                    "HIT" if cache_hit else "MISS",
                    elapsed,
                    "resolved" if result is not None else "skipped",
                )
                report(
                    PreparationProgress(
                        PreparationStage.DOWNLOADING_ARTWORK,
                        completed,
                        len(selected),
                        (
                            f"Downloading artwork {completed} / {len(selected)}\n"
                            f"{_album_label(album)}\n"
                            f"cache: {'HIT' if cache_hit else 'MISS'}"
                        ),
                    )
                )
        download_elapsed = perf_counter() - download_started
        results = tuple(results_by_index[index] for index in sorted(results_by_index))
        logger.debug(
            "Artwork summary: resolution %.2fs, download/cache %.2fs, "
            "cache_hits=%d, cache_misses=%d, unresolved=%d/%d",
            resolution_elapsed,
            download_elapsed,
            cache_hits,
            cache_misses,
            total - len(results),
            total,
        )
        return results

    def _resolve_one(
        self,
        album: Album,
        index: int,
        total: int,
        report: PreparationProgressReporter,
    ) -> ArtworkReference | None:
        report(
            PreparationProgress(
                PreparationStage.RESOLVING_ARTWORK,
                0,
                total,
                f"Finding artwork for album {index + 1} / {total}\n{_album_label(album)}",
            )
        )
        logger.debug("[%d/%d] Resolving %s", index + 1, total, _album_label(album))
        for resolver in self.resolvers:
            try:
                reference = resolver.resolve(album)
                if reference is None:
                    continue
            except (ArtworkResolutionError, OSError) as error:
                logger.debug(
                    "Artwork resolver skipped %s: %s", _album_label(album), error
                )
                continue
            return reference
        return None

    def _cache_one(
        self,
        selected: tuple[Album, ArtworkReference],
    ) -> tuple[ResolvedArtwork | None, bool, float]:
        album, reference = selected
        started = perf_counter()
        cached = self.cache.cached_path(album, reference)
        if cached is not None:
            return ResolvedArtwork(album, cached), True, perf_counter() - started
        try:
            result = ResolvedArtwork(album, self.cache.get(album, reference))
        except (ArtworkCacheError, OSError):
            result = None
        return result, False, perf_counter() - started


def _candidate_score(album: Album, candidate: dict[object, object]) -> float:
    title = candidate.get("title")
    if not isinstance(title, str):
        return 0.0
    artist_credit = candidate.get("artist-credit")
    candidate_artists: list[str] = []
    if isinstance(artist_credit, list):
        for credit in artist_credit:
            if isinstance(credit, dict):
                name = credit.get("name")
                if isinstance(name, str):
                    candidate_artists.append(name)
    artist = " ".join(album.artists)
    candidate_artist = " ".join(candidate_artists)
    score = (
        45.0
        if title == album.album_name
        else 35.0
        if _normalize(title) == _normalize(album.album_name)
        else 0.0
    )
    score += (
        40.0
        if candidate_artist == artist
        else 30.0
        if _normalize(candidate_artist) == _normalize(artist)
        else 0.0
    )
    if candidate.get("primary-type") == "Album":
        score += 10.0
    first_release_date = candidate.get("first-release-date")
    if (
        album.release_date
        and isinstance(first_release_date, str)
        and first_release_date[:4] == album.release_date[:4]
    ):
        score += 3.0
    api_score = candidate.get("score")
    if isinstance(api_score, int):
        score += max(0, min(api_score, 100)) / 20
    return score


def _normalize(value: str) -> str:
    normalized = unicode_normalize("NFKC", value).casefold()
    for original, replacement in (
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u2013", "-"),
        ("\u2014", "-"),
    ):
        normalized = normalized.replace(original, replacement)
    return " ".join(sub(r"[^\w]+", " ", normalized).split())


def _lucene(value: str) -> str:
    return sub(r'([+\-!(){}\[\]^"~*?:\\/])', r"\\\1", value)


def _mapping_key(identity: AlbumKey) -> str:
    return "\x1f".join(identity)


def _album_label(album: Album) -> str:
    artists = ", ".join(album.artists) or "Unknown artist"
    return f'"{album.album_name}" — {artists}'


def _cover_art_reference(
    mbid: str,
    artwork_url: str | None = None,
) -> ArtworkReference:
    url = artwork_url or (
        f"https://{_COVER_ART_HOST}/release-group/{quote(mbid)}/front-500"
    )
    parsed = urlparse(url)
    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    if parsed.scheme not in {"http", "https"} or not any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in _ARCHIVE_HOSTS
    ):
        raise ArtworkResolutionError("Cover Art Archive returned an unsafe image URL")
    secure_url = parsed._replace(scheme="https").geturl()
    return ArtworkReference(
        provider="cover-art-archive",
        identifier=mbid,
        url=secure_url,
    )
