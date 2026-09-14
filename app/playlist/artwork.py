"""Concurrent, validated, disk-backed album artwork retrieval."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from json import dumps
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError

from app.playlist.models import Album, deduplicate_albums

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_RETRIES = 2
_DEFAULT_MAX_WORKERS = 4
_MAX_WORKERS = 16
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
_DEFAULT_MAX_ARTWORK_PIXELS = 25_000_000
_RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class ArtworkCacheError(RuntimeError):
    """Base error for artwork download and validation failures."""


class ArtworkDownloadError(ArtworkCacheError):
    """Raised when artwork cannot be downloaded successfully."""


class ArtworkValidationError(ArtworkCacheError):
    """Raised when downloaded bytes are not valid square image artwork."""


@dataclass(frozen=True, slots=True)
class ArtworkCacheMetadata:
    """Inspectable metadata recorded beside each cached artwork file."""

    cache_key: str
    album_id: str | None
    album_name: str
    artists: tuple[str, ...]
    artwork_url: str
    content_type: str
    content_sha256: str
    width: int
    height: int
    artwork_path: str


class ArtworkCache:
    """Download and retain validated square album artwork as RGB PNG files."""

    def __init__(
        self,
        cache_root: str | Path = "cache",
        *,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        retries: int = _DEFAULT_RETRIES,
        retry_backoff: float = 0.25,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        max_download_bytes: int = _MAX_DOWNLOAD_BYTES,
        max_artwork_pixels: int = _DEFAULT_MAX_ARTWORK_PIXELS,
    ) -> None:
        if timeout <= 0:
            raise ValueError("Artwork timeout must be positive")
        if retries < 0:
            raise ValueError("Artwork retries cannot be negative")
        if retry_backoff < 0:
            raise ValueError("Artwork retry backoff cannot be negative")
        if not 1 <= max_workers <= _MAX_WORKERS:
            raise ValueError(
                f"Artwork concurrency must be between 1 and {_MAX_WORKERS}"
            )
        if max_download_bytes <= 0:
            raise ValueError("Maximum artwork download size must be positive")
        if max_artwork_pixels <= 0:
            raise ValueError("Maximum artwork pixel count must be positive")

        self.cache_root = Path(cache_root)
        self.artwork_dir = self.cache_root / "artwork"
        self.metadata_dir = self.cache_root / "metadata"
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.max_workers = max_workers
        self.max_download_bytes = max_download_bytes
        self.max_artwork_pixels = max_artwork_pixels
        self.artwork_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, Lock] = {}
        self._locks_guard = Lock()

    def cache_key(self, album: Album) -> str:
        """Return a stable filesystem-safe key from the album identity."""
        identity = "\x1f".join(album.identity_key)
        return sha256(identity.encode("utf-8")).hexdigest()

    def get(self, album: Album) -> Path:
        """Return cached artwork, downloading and validating it when absent."""
        cache_key = self.cache_key(album)
        artwork_path = self.artwork_dir / f"{cache_key}.png"
        album_lock = self._lock_for(cache_key)
        with album_lock:
            if _valid_cached_artwork(artwork_path, self.max_artwork_pixels):
                return artwork_path

            artwork_url = album.artwork_url
            if artwork_url is None:
                raise ArtworkDownloadError(
                    f"Album has no artwork URL: {album.album_name}"
                )
            payload, content_type = self._download(artwork_url)
            image = _decode_square_rgb(payload, album, self.max_artwork_pixels)
            metadata = ArtworkCacheMetadata(
                cache_key=cache_key,
                album_id=album.album_id,
                album_name=album.album_name,
                artists=album.artists,
                artwork_url=artwork_url,
                content_type=content_type,
                content_sha256=sha256(payload).hexdigest(),
                width=image.width,
                height=image.height,
                artwork_path=str(artwork_path),
            )
            metadata_path = self.metadata_dir / f"{cache_key}.json"
            try:
                _save_image_atomically(image, artwork_path)
                _save_metadata_atomically(metadata, metadata_path)
            except Exception:
                artwork_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                raise
            return artwork_path

    def get_many(self, albums: Sequence[Album]) -> tuple[Path, ...]:
        """Cache unique albums concurrently while retaining playlist order."""
        unique_albums = deduplicate_albums(albums)
        if not unique_albums:
            return ()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return tuple(executor.map(self.get, unique_albums))

    def _download(self, artwork_url: str) -> tuple[bytes, str]:
        parsed_url = urlparse(artwork_url)
        if (
            parsed_url.scheme.casefold() not in {"http", "https"}
            or not parsed_url.hostname
        ):
            raise ArtworkDownloadError(
                f"Artwork URL must use HTTP or HTTPS: {artwork_url}"
            )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._download_once(artwork_url)
            except HTTPError as error:
                if error.code not in _RETRYABLE_HTTP_STATUSES:
                    raise ArtworkDownloadError(
                        f"Artwork request returned HTTP {error.code}: {artwork_url}"
                    ) from error
                last_error = error
            except (TimeoutError, URLError, OSError) as error:
                last_error = error

            if attempt < self.retries:
                sleep(self.retry_backoff * (2**attempt))

        raise ArtworkDownloadError(
            f"Artwork download failed after {self.retries + 1} attempts: "
            f"{artwork_url} ({last_error})"
        ) from last_error

    def _download_once(self, artwork_url: str) -> tuple[bytes, str]:
        request = Request(
            artwork_url,
            headers={"User-Agent": "Albumosaic/0.1"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            status = response.getcode()
            if status is None or not 200 <= status < 300:
                if status in _RETRYABLE_HTTP_STATUSES:
                    raise HTTPError(
                        artwork_url,
                        status,
                        "retryable artwork response",
                        response.headers,
                        None,
                    )
                raise ArtworkDownloadError(
                    f"Artwork request returned HTTP {status}: {artwork_url}"
                )

            content_type = response.headers.get_content_type().casefold()
            if not content_type.startswith("image/"):
                raise ArtworkValidationError(
                    f"Artwork response is not an image ({content_type}): {artwork_url}"
                )
            payload = response.read(self.max_download_bytes + 1)
            if len(payload) > self.max_download_bytes:
                raise ArtworkValidationError(
                    f"Artwork exceeds {self.max_download_bytes} bytes: {artwork_url}"
                )
            if not payload:
                raise ArtworkValidationError(
                    f"Artwork response is empty: {artwork_url}"
                )
            return payload, content_type

    def _lock_for(self, cache_key: str) -> Lock:
        with self._locks_guard:
            return self._locks.setdefault(cache_key, Lock())


def cache_artwork(
    albums: Sequence[Album],
    cache_dir: Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    retries: int = _DEFAULT_RETRIES,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> tuple[Path, ...]:
    """Convenience wrapper returning local paths for unique album artwork."""
    return ArtworkCache(
        cache_root=cache_dir,
        timeout=timeout,
        retries=retries,
        max_workers=max_workers,
    ).get_many(albums)


def _decode_square_rgb(
    payload: bytes,
    album: Album,
    max_artwork_pixels: int,
) -> Image.Image:
    try:
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            if source.width * source.height > max_artwork_pixels:
                raise ArtworkValidationError(
                    f"Artwork exceeds {max_artwork_pixels} pixels for album "
                    f"{album.album_name}"
                )
            source.load()
            if source.width != source.height:
                raise ArtworkValidationError(
                    f"Artwork is not square for album {album.album_name}: "
                    f"{source.width}x{source.height}"
                )
            return source.convert("RGB")
    except ArtworkValidationError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ArtworkValidationError(
            f"Pillow could not decode artwork for album {album.album_name}"
        ) from error


def _valid_cached_artwork(path: Path, max_artwork_pixels: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            if image.width * image.height > max_artwork_pixels:
                return False
            image.load()
            return (
                image.mode == "RGB" and image.width > 0 and image.width == image.height
            )
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        return False


def _save_image_atomically(image: Image.Image, destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.stem}-",
            suffix=".png",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            image.save(temporary, format="PNG")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _save_metadata_atomically(
    metadata: ArtworkCacheMetadata,
    destination: Path,
) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.stem}-",
            suffix=".json",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(dumps(asdict(metadata), indent=2, sort_keys=True))
            temporary.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
