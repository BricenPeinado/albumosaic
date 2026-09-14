"""Independent local and MusicBrainz artwork resolution strategies."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from json import dumps, loads
from pathlib import Path
from re import sub
from threading import Lock
from time import monotonic, sleep
from typing import Protocol, cast
from unicodedata import normalize as unicode_normalize
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from app.playlist.artwork import ArtworkCache, ArtworkCacheError
from app.playlist.models import Album, AlbumKey, deduplicate_albums

_MUSICBRAINZ_HOST = "musicbrainz.org"
_COVER_ART_HOST = "coverartarchive.org"
_ARCHIVE_HOSTS = frozenset({_COVER_ART_HOST, "archive.org"})


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
        timeout: float = 15.0,
        minimum_interval: float = 1.0,
        confidence_threshold: float = 75.0,
    ) -> None:
        if not contact.strip():
            raise ValueError("MusicBrainz contact information is required")
        if timeout <= 0 or minimum_interval < 0:
            raise ValueError("MusicBrainz timing values are invalid")
        self.user_agent = f"Albumosaic/0.1 ({contact.strip()})"
        self.cache_path = Path(cache_path)
        self.timeout = timeout
        self.minimum_interval = minimum_interval
        self.confidence_threshold = confidence_threshold
        self._lock = Lock()
        self._last_request = 0.0
        self._mapping = self._load_mapping()

    def resolve(self, album: Album) -> ArtworkReference | None:
        key = _mapping_key(album.identity_key)
        if key in self._mapping:
            mbid = self._mapping[key]
            return self._cover_reference(mbid) if mbid else None

        artist = " ".join(album.artists)
        query = (
            f'releasegroup:"{_lucene(album.album_name)}" AND artist:"{_lucene(artist)}"'
        )
        url = f"https://{_MUSICBRAINZ_HOST}/ws/2/release-group/?" + urlencode(
            {"query": query, "fmt": "json", "limit": 10}
        )
        payload = self._get_json(url)
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
        reference = self._cover_reference(mbid)
        self._remember(key, mbid if reference is not None else None)
        return reference

    def _cover_reference(self, mbid: str) -> ArtworkReference | None:
        url = f"https://{_COVER_ART_HOST}/release-group/{quote(mbid)}/front-500"
        request = Request(url, headers={"User-Agent": self.user_agent}, method="HEAD")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                final_url = response.geturl()
        except HTTPError as error:
            if error.code == 404:
                return None
            raise ArtworkResolutionError(
                f"Cover Art Archive returned HTTP {error.code}"
            ) from error
        except (TimeoutError, URLError, OSError) as error:
            raise ArtworkResolutionError("Cover Art Archive request failed") from error
        final_host = urlparse(final_url).hostname
        if final_host not in _ARCHIVE_HOSTS:
            raise ArtworkResolutionError(
                "Cover Art Archive redirected to an unsafe host"
            )
        return ArtworkReference(provider="cover-art-archive", identifier=mbid, url=url)

    def _get_json(self, url: str) -> dict[str, object]:
        if urlparse(url).hostname != _MUSICBRAINZ_HOST:
            raise ArtworkResolutionError("Refused an unexpected MusicBrainz URL")
        with self._lock:
            delay = self.minimum_interval - (monotonic() - self._last_request)
            if delay > 0:
                sleep(delay)
            request = Request(url, headers={"User-Agent": self.user_agent})
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    decoded = loads(response.read())
            except HTTPError as error:
                raise ArtworkResolutionError(
                    f"MusicBrainz returned HTTP {error.code}"
                ) from error
            except (TimeoutError, URLError, OSError, ValueError) as error:
                raise ArtworkResolutionError("MusicBrainz request failed") from error
            finally:
                self._last_request = monotonic()
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
            str(key): value if isinstance(value, str) else None
            for key, value in decoded.items()
        }

    def _remember(self, key: str, mbid: str | None) -> None:
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

    def get_many(self, albums: tuple[Album, ...]) -> tuple[ResolvedArtwork, ...]:
        selected: list[tuple[Album, ArtworkReference]] = []
        for album in deduplicate_albums(albums):
            for resolver in self.resolvers:
                try:
                    reference = resolver.resolve(album)
                    if reference is None:
                        continue
                except (ArtworkResolutionError, OSError):
                    continue
                selected.append((album, reference))
                break
        with ThreadPoolExecutor(max_workers=min(4, len(selected) or 1)) as executor:
            results = executor.map(self._cache_one, selected)
            return tuple(result for result in results if result is not None)

    def _cache_one(
        self,
        selected: tuple[Album, ArtworkReference],
    ) -> ResolvedArtwork | None:
        album, reference = selected
        try:
            return ResolvedArtwork(album, self.cache.get(album, reference))
        except (ArtworkCacheError, OSError):
            return None


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
