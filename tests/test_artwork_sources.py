"""Tests for independent artwork resolution and provider priority."""

from __future__ import annotations

from json import dumps
from pathlib import Path
from urllib.error import HTTPError

import pytest
from PIL import Image

from app.playlist import artwork_sources
from app.playlist.artwork import ArtworkCache, ArtworkDownloadError
from app.playlist.artwork_sources import (
    ArtworkReference,
    IndependentArtworkProvider,
    LocalManifestArtworkResolver,
    MusicBrainzArtworkResolver,
)
from app.playlist.models import Album


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, object] | None = None,
        *,
        final_url: str = "https://archive.org/download/cover.jpg",
    ) -> None:
        self.payload = dumps(payload or {}).encode()
        self.final_url = final_url

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.payload

    def geturl(self) -> str:
        return self.final_url


def album(name: str = "The Album", artist: str = "The Artist") -> Album:
    return Album(f"{artist}:{name}", name, (artist,), None, None)


def candidate(
    *,
    name: str = "The Album",
    artist: str = "The Artist",
    mbid: str = "release-group-id",
    score: int = 100,
) -> dict[str, object]:
    return {
        "id": mbid,
        "title": name,
        "artist-credit": [{"name": artist}],
        "primary-type": "Album",
        "score": score,
    }


def test_musicbrainz_exact_and_normalized_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searches = iter(
        [
            {"release-groups": [candidate()]},
            {
                "release-groups": [
                    candidate(name="the—album", artist="THE ARTIST", mbid="normalized")
                ]
            },
        ]
    )

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del timeout
        if request.get_method() == "HEAD":  # type: ignore[attr-defined]
            return FakeResponse()
        return FakeResponse(next(searches))

    monkeypatch.setattr(artwork_sources, "urlopen", fake_urlopen)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    exact = resolver.resolve(album())
    normalized = resolver.resolve(album("The Album", "the artist"))

    assert exact is not None and exact.identifier == "release-group-id"
    assert normalized is not None and normalized.identifier == "normalized"


def test_ambiguous_musicbrainz_match_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "release-groups": [
            candidate(mbid="one"),
            candidate(mbid="two"),
        ]
    }
    monkeypatch.setattr(
        artwork_sources,
        "urlopen",
        lambda request, timeout: FakeResponse(payload),
    )
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    assert resolver.resolve(album()) is None


def test_missing_cover_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del timeout
        if request.get_method() == "HEAD":  # type: ignore[attr-defined]
            raise HTTPError("url", 404, "missing", {}, None)
        return FakeResponse({"release-groups": [candidate()]})

    monkeypatch.setattr(artwork_sources, "urlopen", fake_urlopen)
    resolver = MusicBrainzArtworkResolver(
        "maintainer@example.com",
        tmp_path / "mapping.json",
        minimum_interval=0,
    )

    assert resolver.resolve(album()) is None


def test_local_manifest_takes_priority_and_partial_results_continue(
    tmp_path: Path,
) -> None:
    cover = tmp_path / "cover.png"
    Image.new("RGB", (16, 16), "blue").save(cover)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        dumps([{"artist": "The Artist", "album": "The Album", "path": "cover.png"}]),
        encoding="utf-8",
    )
    local = LocalManifestArtworkResolver(manifest)

    class RemoteResolver:
        calls = 0

        def resolve(self, target: Album) -> ArtworkReference | None:
            del target
            self.calls += 1
            return None

    remote = RemoteResolver()
    provider = IndependentArtworkProvider(
        (local, remote),
        ArtworkCache(tmp_path / "cache", allowed_hosts={"coverartarchive.org"}),
    )
    results = provider.get_many((album(), album("Missing")))

    assert len(results) == 1
    assert results[0].album.album_name == "The Album"
    assert results[0].path.is_file()
    assert remote.calls == 1


def test_spotify_image_hosts_are_rejected_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_call(request: object, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        return FakeResponse()

    monkeypatch.setattr("app.playlist.artwork.urlopen", forbidden_call)
    cache = ArtworkCache(tmp_path, allowed_hosts={"coverartarchive.org", "archive.org"})
    reference = ArtworkReference(
        "forbidden",
        "id",
        url="https://i.scdn.co/image/spotify-cover",
    )

    with pytest.raises(ArtworkDownloadError, match="not permitted"):
        cache.get(album(), reference)
    assert calls == 0
