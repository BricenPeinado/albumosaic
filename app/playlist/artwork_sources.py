"""Independent local and MusicBrainz artwork resolution strategies."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from re import compile as re_compile
from re import sub
from threading import Lock
from time import monotonic, perf_counter, sleep
from typing import Protocol, cast
from unicodedata import normalize as unicode_normalize
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request

from app.network import open_url
from app.playlist.artwork import (
    ArtworkCache,
    ArtworkCacheError,
    ArtworkDownloadError,
    ArtworkValidationError,
)
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
_MATCH_CACHE_VERSION = 2
_MAX_MUSICBRAINZ_SEARCHES = 3
_EDITION_SUFFIX = re_compile(
    r"(?i)(?:\s*[([]\s*|\s*[-\u2013\u2014]\s*)"
    r"(?:deluxe(?:\s+edition)?|expanded\s+edition|remaster(?:ed)?"
    r"(?:\s+\d{4})?|(?:\d+(?:st|nd|rd|th)\s+)?anniversary\s+edition|"
    r"bonus\s+track\s+version|explicit|clean\s+version)\s*[)\]]?\s*$"
)

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
        self._timing_lock = Lock()
        self._mb_calls = 0
        self._mb_seconds = 0.0
        self._caa_calls = 0
        self._caa_seconds = 0.0
        self._last_request = 0.0
        self._mapping, self._reasons, self._contexts, self._cover_refs = (
            self._load_mapping()
        )

    def resolve(self, album: Album) -> ArtworkReference | None:
        label = _album_label(album)
        key = _mapping_key(album.identity_key)
        with self._mapping_lock:
            mapping_cached = key in self._mapping
            cached_mbid = self._mapping.get(key)
            cached_reason = self._reasons.get(key)
            cached_context = self._contexts.get(key)
            cached_cover = self._cover_refs.get(cached_mbid or "")
        if (
            mapping_cached
            and cached_mbid is None
            and cached_context != self._match_context(album)
        ):
            mapping_cached = False
            logger.debug("MusicBrainz mapping cache: stale negative")
        if mapping_cached:
            logger.debug("%s\nMusicBrainz mapping cache: HIT", label)
            if cached_mbid is None:
                logger.debug("%s: cached negative", cached_reason or "low_confidence")
                return None
            if cached_cover is not None:
                logger.debug("CAA cover-reference cache: HIT")
                return _cover_art_reference(cached_mbid, cached_cover)
            logger.debug("CAA cover-reference cache: MISS")
            mbid = cached_mbid
        else:
            logger.debug("%s\nMusicBrainz mapping cache: MISS", label)
            found_mbid = self._search_album(album, key)
            if found_mbid is None:
                return None
            mbid = found_mbid
        cover_started = perf_counter()
        try:
            reference = self._cover_reference(mbid)
        except ArtworkResolutionError as error:
            logger.debug(
                "cover_lookup_error: %s after %.2fs",
                error,
                perf_counter() - cover_started,
            )
            raise
        finally:
            self._record_service_timing("caa", perf_counter() - cover_started)
        logger.debug("CAA lookup: %.2fs", perf_counter() - cover_started)
        if reference is None:
            logger.debug("no_cover_art: %s", label)
            self._remember(key, mbid, reason="no_cover_art")
        else:
            self._remember(key, mbid, cover_url=reference.url)
        return reference

    def _search_album(self, album: Album, key: str) -> str | None:
        candidates_by_id: dict[str, dict[str, object]] = {}
        seen_queries: set[str] = set()
        for stage, query in enumerate(_search_queries(album), start=1):
            if query in seen_queries:
                continue
            seen_queries.add(query)
            url = f"https://{_MUSICBRAINZ_HOST}/ws/2/release-group/?" + urlencode(
                {"query": query, "fmt": "json", "limit": 10}
            )
            started = perf_counter()
            try:
                payload = self._get_json(url)
            except ArtworkResolutionError as error:
                logger.debug(
                    "MusicBrainz search %d: %s after %.2fs",
                    stage,
                    error,
                    perf_counter() - started,
                )
                raise
            finally:
                self._record_service_timing("mb", perf_counter() - started)
            logger.debug(
                "MusicBrainz search %d: %.2fs", stage, perf_counter() - started
            )
            results = payload.get("release-groups")
            if not isinstance(results, list):
                raise ArtworkResolutionError(
                    "MusicBrainz returned invalid search results"
                )
            for item in results:
                if isinstance(item, dict):
                    mbid = item.get("id")
                    if isinstance(mbid, str) and mbid:
                        candidates_by_id[mbid] = item
            winner, reason = _select_candidate(
                album, candidates_by_id.values(), self.confidence_threshold
            )
            if winner is not None:
                mbid = winner["id"]
                assert isinstance(mbid, str)
                self._remember(key, mbid)
                return mbid
            logger.debug("MusicBrainz search %d: %s", stage, reason)

        _, reason = _select_candidate(
            album, candidates_by_id.values(), self.confidence_threshold
        )
        logger.debug("%s: %s", reason, _album_label(album))
        self._remember(
            key, None, reason=reason, match_context=self._match_context(album)
        )
        return None

    def _match_context(self, album: Album) -> str:
        values = (
            _normalize(album.album_name),
            _normalize(album.artists[0]) if album.artists else "",
            str(_year(album.release_date) or ""),
            (album.album_type or "album").casefold(),
            str(self.confidence_threshold),
        )
        return sha256("\x1f".join(values).encode("utf-8")).hexdigest()

    def service_timing_snapshot(self) -> tuple[int, float, int, float]:
        """Return cumulative request counts and wall times for diagnostics."""
        with self._timing_lock:
            return (
                self._mb_calls,
                self._mb_seconds,
                self._caa_calls,
                self._caa_seconds,
            )

    def _record_service_timing(self, service: str, elapsed: float) -> None:
        with self._timing_lock:
            if service == "mb":
                self._mb_calls += 1
                self._mb_seconds += elapsed
            else:
                self._caa_calls += 1
                self._caa_seconds += elapsed

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

    def _load_mapping(
        self,
    ) -> tuple[dict[str, str | None], dict[str, str], dict[str, str], dict[str, str]]:
        if not self.cache_path.is_file():
            return {}, {}, {}, {}
        try:
            decoded = loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}, {}, {}
        if not isinstance(decoded, dict):
            return {}, {}, {}, {}
        if decoded.get("version") != _MATCH_CACHE_VERSION:
            # Legacy positive mappings are useful; legacy nulls predate this strategy.
            if "version" in decoded:
                return {}, {}, {}, {}
            positives: dict[str, str | None] = {
                str(key): value
                for key, value in decoded.items()
                if isinstance(value, str) and value
            }
            return positives, {}, {}, {}
        raw_entries = decoded.get("entries")
        raw_covers = decoded.get("covers")
        if not isinstance(raw_entries, dict):
            return {}, {}, {}, {}
        mapping: dict[str, str | None] = {}
        reasons: dict[str, str] = {}
        contexts: dict[str, str] = {}
        for key, entry in raw_entries.items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            mbid = entry.get("mbid")
            if mbid is not None and (not isinstance(mbid, str) or not mbid):
                continue
            reason = entry.get("reason")
            context = entry.get("match_context")
            if mbid is None and reason not in {
                "no_musicbrainz_results",
                "low_confidence",
                "ambiguous_match",
            }:
                continue
            if mbid is None and not isinstance(context, str):
                continue
            mapping[key] = mbid
            if isinstance(reason, str):
                reasons[key] = reason
            if isinstance(context, str):
                contexts[key] = context
        covers: dict[str, str] = {}
        if isinstance(raw_covers, dict):
            for mbid, url in raw_covers.items():
                if isinstance(mbid, str) and isinstance(url, str):
                    try:
                        _cover_art_reference(mbid, url)
                    except ArtworkResolutionError:
                        continue
                    covers[mbid] = url
        return mapping, reasons, contexts, covers

    def _remember(
        self,
        key: str,
        mbid: str | None,
        *,
        reason: str | None = None,
        cover_url: str | None = None,
        match_context: str | None = None,
    ) -> None:
        with self._mapping_lock:
            self._mapping[key] = mbid
            if reason is None:
                self._reasons.pop(key, None)
            else:
                self._reasons[key] = reason
            if match_context is None:
                self._contexts.pop(key, None)
            else:
                self._contexts[key] = match_context
            if mbid is not None and cover_url is not None:
                self._cover_refs[mbid] = cover_url
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.cache_path.with_suffix(".tmp")
                payload = {
                    "version": _MATCH_CACHE_VERSION,
                    "entries": {
                        identity: {
                            "mbid": value,
                            "reason": self._reasons.get(identity),
                            "match_context": self._contexts.get(identity),
                        }
                        for identity, value in self._mapping.items()
                    },
                    "covers": self._cover_refs,
                }
                temporary.write_text(dumps(payload, indent=2), encoding="utf-8")
                temporary.replace(self.cache_path)
            except OSError as error:
                logger.debug("MusicBrainz cache write failed: %s", error)
            finally:
                with suppress(OSError):
                    self.cache_path.with_suffix(".tmp").unlink(missing_ok=True)


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
        service_baselines = {
            resolver: resolver.service_timing_snapshot()
            for resolver in self.resolvers
            if isinstance(resolver, MusicBrainzArtworkResolver)
        }
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
        for resolver, baseline in service_baselines.items():
            current = resolver.service_timing_snapshot()
            logger.debug(
                "Artwork service timing: MusicBrainz %d searches / %.3fs cumulative; "
                "CAA %d metadata lookups / %.3fs cumulative; "
                "artwork download/cache %.3fs wall",
                current[0] - baseline[0],
                current[1] - baseline[1],
                current[2] - baseline[2],
                current[3] - baseline[3],
                download_elapsed,
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
        except ArtworkValidationError as error:
            logger.debug("artwork_validation_error: %s: %s", _album_label(album), error)
            result = None
        except (ArtworkDownloadError, OSError) as error:
            logger.debug("artwork_download_error: %s: %s", _album_label(album), error)
            result = None
        except ArtworkCacheError as error:
            logger.debug("artwork_download_error: %s: %s", _album_label(album), error)
            result = None
        return result, False, perf_counter() - started


def _search_queries(album: Album) -> tuple[str, ...]:
    primary_artist = album.artists[0] if album.artists else ""
    full_title = album.album_name.strip()
    fallback_title = _edition_title(full_title)
    exact = (
        f'releasegroup:"{_lucene(full_title)}" AND artist:"{_lucene(primary_artist)}"'
    )
    queries = [exact]
    if _normalize(fallback_title) != _normalize(full_title):
        queries.append(
            f'releasegroup:"{_lucene(fallback_title)}" AND '
            f'artist:"{_lucene(primary_artist)}"'
        )
    queries.append(
        f"releasegroup:({_lucene(fallback_title)}) AND "
        f"artist:({_lucene(primary_artist)})"
    )
    return tuple(queries[:_MAX_MUSICBRAINZ_SEARCHES])


def _select_candidate(
    album: Album,
    candidates: Iterable[dict[str, object]],
    confidence_threshold: float,
) -> tuple[dict[str, object] | None, str]:
    ranked = sorted(
        ((_candidate_score(album, candidate), candidate) for candidate in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not ranked:
        return None, "no_musicbrainz_results"
    if ranked[0][0] < confidence_threshold:
        return None, "low_confidence"
    if (
        len(ranked) > 1
        and ranked[0][0] - ranked[1][0] < 5
        and not _same_effective_album(ranked[0][1], ranked[1][1])
    ):
        return None, "ambiguous_match"
    return ranked[0][1], "resolved"


def _candidate_score(album: Album, candidate: dict[str, object]) -> float:
    title = candidate.get("title")
    if not isinstance(title, str):
        return 0.0
    target_title = _normalize(album.album_name)
    candidate_title = _normalize(title)
    if title == album.album_name:
        score = 40.0
    elif candidate_title == target_title:
        score = 36.0
    elif _normalize(_edition_title(title)) == _normalize(
        _edition_title(album.album_name)
    ):
        score = 32.0
    else:
        score = 34.0 * SequenceMatcher(None, target_title, candidate_title).ratio()

    primary_artist = album.artists[0] if album.artists else ""
    candidate_artist = _candidate_artist(candidate)
    target_artist = _normalize(primary_artist)
    matched_artist = _normalize(candidate_artist)
    if candidate_artist == primary_artist:
        score += 35.0
    elif matched_artist == target_artist:
        score += 32.0
    else:
        score += 32.0 * SequenceMatcher(None, target_artist, matched_artist).ratio()

    album_type = (album.album_type or "album").casefold()
    primary_type = candidate.get("primary-type")
    if isinstance(primary_type, str):
        if primary_type.casefold() == album_type:
            score += 8.0
        elif album_type == "compilation" and primary_type.casefold() == "album":
            secondary = candidate.get("secondary-types")
            if isinstance(secondary, list) and "Compilation" in secondary:
                score += 8.0
        else:
            score -= 5.0

    target_year = _year(album.release_date)
    candidate_year = _year(candidate.get("first-release-date"))
    if target_year is not None and candidate_year is not None:
        difference = abs(target_year - candidate_year)
        score += (
            10.0
            if difference == 0
            else 7.0
            if difference == 1
            else 4.0
            if difference <= 3
            else -10.0
        )

    api_score = candidate.get("score")
    if isinstance(api_score, int):
        score += max(0, min(api_score, 100)) / 25
    return score


def _candidate_artist(candidate: dict[str, object]) -> str:
    credits = candidate.get("artist-credit")
    if isinstance(credits, list):
        for credit in credits:
            if isinstance(credit, dict):
                name = credit.get("name")
                if isinstance(name, str):
                    return name
    return ""


def _same_effective_album(left: dict[str, object], right: dict[str, object]) -> bool:
    left_title = left.get("title")
    right_title = right.get("title")
    left_year = _year(left.get("first-release-date"))
    right_year = _year(right.get("first-release-date"))
    return (
        isinstance(left_title, str)
        and isinstance(right_title, str)
        and _normalize(_edition_title(left_title))
        == _normalize(_edition_title(right_title))
        and _normalize(_candidate_artist(left)) == _normalize(_candidate_artist(right))
        and left_year is not None
        and left_year == right_year
    )


def _edition_title(title: str) -> str:
    cleaned = title.strip()
    while True:
        shorter = _EDITION_SUFFIX.sub("", cleaned).strip()
        if shorter == cleaned or not shorter:
            return cleaned
        cleaned = shorter


def _year(value: object) -> int | None:
    if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None


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
