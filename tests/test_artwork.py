"""Tests for concurrent, validated album artwork caching."""

from __future__ import annotations

from email.message import Message
from io import BytesIO
from json import loads
from pathlib import Path
from threading import Lock
from time import sleep
from urllib.error import URLError

import pytest
from PIL import Image

from app.playlist import artwork
from app.playlist.artwork import (
    ArtworkCache,
    ArtworkDownloadError,
    ArtworkValidationError,
)
from app.playlist.artwork_sources import ArtworkReference
from app.playlist.models import Album


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        content_type: str = "image/png",
        status: int = 200,
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def getcode(self) -> int:
        return self.status

    def read(self, limit: int = -1) -> bytes:
        return self.payload if limit < 0 else self.payload[:limit]


def make_album(
    album_id: str | None = "album-1",
    *,
    album_name: str = "Album",
    artist: str = "Artist",
) -> Album:
    return Album(
        album_id=album_id,
        album_name=album_name,
        artists=(artist,),
        source_url=None,
    )


def remote_reference(
    url: str = "https://images.example/cover.png",
) -> ArtworkReference:
    return ArtworkReference("test", url, url=url)


def image_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, (20, 40, 60, 128)).save(output, format="PNG")
    return output.getvalue()


def test_download_validates_and_stores_original_square_rgb_artwork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_timeouts: list[float] = []

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del request
        requested_timeouts.append(timeout)
        return FakeResponse(image_bytes())

    monkeypatch.setattr(artwork, "urlopen", fake_urlopen)
    cache = ArtworkCache(tmp_path, timeout=1.5)

    path = cache.get(make_album(), remote_reference())

    assert path.parent == tmp_path / "artwork"
    assert path.suffix == ".png"
    assert requested_timeouts == [1.5]
    with Image.open(path) as cached_image:
        assert cached_image.mode == "RGB"
        assert cached_image.size == (32, 32)

    metadata_path = tmp_path / "metadata" / f"{path.stem}.json"
    metadata = loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["album_id"] == "album-1"
    assert metadata["content_type"] == "image/png"
    assert metadata["artwork_origin"] == "https://images.example/cover.png"
    assert metadata["width"] == 32
    assert len(metadata["content_sha256"]) == 64


def test_valid_cached_artwork_is_never_redownloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse(image_bytes())

    monkeypatch.setattr(artwork, "urlopen", fake_urlopen)
    cache = ArtworkCache(tmp_path)

    reference = remote_reference()
    first = cache.get(make_album(), reference)
    second = cache.get(make_album(album_name="Renamed"), reference)

    assert first == second
    assert calls == 1


def test_cache_key_uses_normalized_fallback_without_album_id(tmp_path: Path) -> None:
    cache = ArtworkCache(tmp_path)
    first = make_album(None, album_name="  HOME ", artist="THE ARTIST")
    second = make_album(None, album_name="home", artist="the artist")

    assert cache.cache_key(first) == cache.cache_key(second)


def test_transient_download_failures_are_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flaky_urlopen(request: object, timeout: float) -> FakeResponse:
        nonlocal attempts
        del request, timeout
        attempts += 1
        if attempts < 3:
            raise URLError("temporary failure")
        return FakeResponse(image_bytes())

    monkeypatch.setattr(artwork, "urlopen", flaky_urlopen)
    cache = ArtworkCache(tmp_path, retries=2, retry_backoff=0)

    assert cache.get(make_album(), remote_reference()).is_file()
    assert attempts == 3


@pytest.mark.parametrize(
    ("payload", "content_type", "message"),
    [
        (b"not an image", "image/png", "decode"),
        (image_bytes(), "text/html", "not an image"),
        (image_bytes((40, 20)), "image/png", "not square"),
    ],
)
def test_malformed_artwork_is_rejected_without_cache_files(
    payload: bytes,
    content_type: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artwork,
        "urlopen",
        lambda request, timeout: FakeResponse(payload, content_type=content_type),
    )
    cache = ArtworkCache(tmp_path)

    with pytest.raises(ArtworkValidationError, match=message):
        cache.get(make_album(), remote_reference())

    assert not list((tmp_path / "artwork").glob("*.png"))
    assert not list((tmp_path / "metadata").glob("*.json"))


def test_non_http_artwork_url_is_rejected(tmp_path: Path) -> None:
    cache = ArtworkCache(tmp_path)

    with pytest.raises(ArtworkDownloadError, match="HTTP or HTTPS"):
        cache.get(make_album(), remote_reference("file:///tmp/cover.png"))


def test_get_many_deduplicates_albums_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artwork,
        "urlopen",
        lambda request, timeout: FakeResponse(image_bytes()),
    )
    first = make_album("album-1")
    duplicate = make_album("album-1")
    second = make_album("album-2")

    paths = ArtworkCache(tmp_path).get_many(
        (
            (first, remote_reference("https://images.example/1.png")),
            (duplicate, remote_reference("https://images.example/1.png")),
            (second, remote_reference("https://images.example/2.png")),
        )
    )

    assert len(paths) == 2
    assert paths[0] != paths[1]


def test_concurrent_downloads_respect_worker_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_lock = Lock()
    active = 0
    maximum_active = 0

    class MeasuredResponse(FakeResponse):
        def __enter__(self) -> MeasuredResponse:
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            return self

        def read(self, limit: int = -1) -> bytes:
            sleep(0.02)
            return super().read(limit)

        def __exit__(self, *args: object) -> None:
            nonlocal active
            del args
            with state_lock:
                active -= 1

    monkeypatch.setattr(
        artwork,
        "urlopen",
        lambda request, timeout: MeasuredResponse(image_bytes()),
    )
    references = tuple(
        (
            make_album(f"album-{index}"),
            remote_reference(f"https://images.example/{index}.png"),
        )
        for index in range(8)
    )

    paths = ArtworkCache(tmp_path, max_workers=3).get_many(references)

    assert len(paths) == 8
    assert 1 < maximum_active <= 3


def test_concurrency_limit_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 and 16"):
        ArtworkCache(tmp_path, max_workers=17)


def test_artwork_pixel_limit_rejects_decompression_risk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artwork,
        "urlopen",
        lambda request, timeout: FakeResponse(image_bytes((32, 32))),
    )
    cache = ArtworkCache(tmp_path, max_artwork_pixels=100)

    with pytest.raises(ArtworkValidationError, match="exceeds 100 pixels"):
        cache.get(make_album(), remote_reference())


def test_artwork_limits_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pixel count must be positive"):
        ArtworkCache(tmp_path, max_artwork_pixels=0)
